#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""https://github.com/Nervengift/read_thermal.py"""

import io
import json
import os
import re
import subprocess

import numpy as np
from matplotlib import cm
from PIL import Image, UnidentifiedImageError

# ruff: noqa


class FlirExifNotFoundError(ValueError):
  pass


class FlirImageExtractor:
  def __init__(self, exiftool_path='exiftool', is_debug=False):
    self.exiftool_path = exiftool_path
    self.is_debug = is_debug
    self.flir_img_filename = ''
    self.image_suffix = '_rgb_image.jpg'
    self.thumbnail_suffix = '_rgb_thumb.jpg'
    self.thermal_suffix = '_thermal.png'
    self.default_distance = 1.0

    # valid for PNG thermal images
    self.use_thumbnail = False
    self.fix_endian = True

    self.rgb_image_np = None
    self.thermal_image_np = None

    self._raw2temp_vec = None

  def process_image(self, flir_img_filename):
    """
    Given a valid image path, process the file: extract real thermal values
    and a thumbnail for comparison (generally thumbnail is on the visible spectre)
    :param flir_img_filename:
    :return:
    """
    if self.is_debug:
      print('INFO Flir image filepath:{}'.format(flir_img_filename))

    if not os.path.isfile(flir_img_filename):
      raise ValueError(
        "Input file does not exist or this user don't have permission on this file"
      )

    self.flir_img_filename = flir_img_filename

    if self.get_image_type().upper().strip() == 'TIFF':
      # valid for tiff images from Zenmuse XTR
      self.use_thumbnail = True
      self.fix_endian = False

    self.rgb_image_np = self.extract_embedded_image()
    self.thermal_image_np = self.extract_thermal_image()

  def get_image_type(self):
    """
    Get the embedded thermal image type, generally can be TIFF or PNG
    :return:
    """
    key = 'RawThermalImageType'
    args = [
      self.exiftool_path,
      ('-' + key),
      '-j',
      self.flir_img_filename,
    ]
    meta_json = subprocess.check_output(args, stderr=subprocess.DEVNULL)
    meta = json.loads(meta_json.decode())[0]

    if key not in meta:
      raise FlirExifNotFoundError(f'{key} not found in {self.flir_img_filename}')

    return meta[key]

  def get_rgb_np(self):
    """
    Return the last extracted rgb image
    :return:
    """
    return self.rgb_image_np

  def get_thermal_np(self):
    """
    Return the last extracted thermal image
    :return:
    """
    return self.thermal_image_np

  def extract_embedded_image(self):
    """
    extracts the visual image as 2D numpy array of RGB values
    """
    image_tag = '-EmbeddedImage'
    # ThumbnailImage 대신 적용
    if self.use_thumbnail:
      image_tag = '-ThumbnailImage'

    args = [
      self.exiftool_path,
      image_tag,
      '-b',
      self.flir_img_filename,
    ]
    visual_img_bytes = subprocess.check_output(args, stderr=subprocess.DEVNULL)
    visual_img_stream = io.BytesIO(visual_img_bytes)

    try:
      visual_img = Image.open(visual_img_stream)
      visual_np = np.array(visual_img)
    except UnidentifiedImageError:
      visual_np = None

    return visual_np

  def extract_thermal_image(self):
    """
    extracts the thermal image as 2D numpy array with temperatures in oC
    """

    # read image metadata needed for conversion of the raw sensor values
    # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,
    # PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
    meta_args = [
      self.exiftool_path,
      self.flir_img_filename,
      '-Emissivity',
      '-SubjectDistance',
      '-AtmosphericTemperature',
      '-ReflectedApparentTemperature',
      '-IRWindowTemperature',
      '-IRWindowTransmission',
      '-RelativeHumidity',
      '-PlanckR1',
      '-PlanckB',
      '-PlanckF',
      '-PlanckO',
      '-PlanckR2',
      '-j',
    ]
    meta_json = subprocess.check_output(meta_args, stderr=subprocess.DEVNULL)
    meta = json.loads(meta_json.decode())[0]

    # exifread can't extract the embedded thermal image, use exiftool instead
    img_args = [
      self.exiftool_path,
      '-RawThermalImage',
      '-b',
      self.flir_img_filename,
    ]
    thermal_img_bytes = subprocess.check_output(img_args, stderr=subprocess.DEVNULL)
    thermal_img_stream = io.BytesIO(thermal_img_bytes)

    thermal_img = Image.open(thermal_img_stream)
    thermal_np = np.array(thermal_img)

    # raw values -> temperature
    subject_distance = self.default_distance
    if 'SubjectDistance' in meta:
      subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

    if self.fix_endian:
      # fix endianness, the bytes in the embedded png are in the wrong order
      thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00FF) << 8))(thermal_np)

    thermal_np = self.raw2temp(
      thermal_np,
      E=meta['Emissivity'],
      OD=subject_distance,
      RTemp=FlirImageExtractor.extract_float(meta['ReflectedApparentTemperature']),
      ATemp=FlirImageExtractor.extract_float(meta['AtmosphericTemperature']),
      IRWTemp=FlirImageExtractor.extract_float(meta['IRWindowTemperature']),
      IRT=meta['IRWindowTransmission'],
      RH=FlirImageExtractor.extract_float(meta['RelativeHumidity']),
      PR1=meta['PlanckR1'],
      PB=meta['PlanckB'],
      PF=meta['PlanckF'],
      PO=meta['PlanckO'],
      PR2=meta['PlanckR2'],
    )

    return thermal_np

  @staticmethod
  def raw2temp(
    raw,
    E=1,
    OD=1,
    RTemp=20,
    ATemp=20,
    IRWTemp=20,
    IRT=1,
    RH=50,
    PR1=21106.77,
    PB=1501,
    PF=1,
    PO=-7340,
    PR2=0.012545258,
  ):
    """
    convert raw values from the flir sensor to temperatures in C
    # this calculation has been ported to python from
    # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
    # a detailed explanation of what is going on here can be found there
    """

    # constants
    ATA1 = 0.006569
    ATA2 = 0.01262
    ATB1 = -0.002276
    ATB2 = -0.00667
    ATX = 1.9

    # transmission through window (calibrated)
    emiss_wind = 1 - IRT
    refl_wind = 0

    # transmission through the air
    h2o = (RH / 100) * np.exp(
      1.5587
      + 0.06939 * (ATemp)
      - 0.00027816 * (ATemp) ** 2
      + 0.00000068455 * (ATemp) ** 3
    )

    tau1 = ATX * np.exp(-np.sqrt(OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (
      1 - ATX
    ) * np.exp(-np.sqrt(OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))

    tau2 = ATX * np.exp(-np.sqrt(OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (
      1 - ATX
    ) * np.exp(-np.sqrt(OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))

    # radiance from the environment
    raw_refl1 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl1_attn = (1 - E) / E * raw_refl1

    raw_atm1 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1

    raw_wind = PR1 / (PR2 * (np.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
    raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind

    raw_refl2 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2

    raw_atm2 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

    raw_obj = (
      raw / E / tau1 / IRT / tau2
      - raw_atm1_attn
      - raw_atm2_attn
      - raw_wind_attn
      - raw_refl1_attn
      - raw_refl2_attn
    )

    # temperature from radiance
    RSob = PR1 / (PR2 * (raw_obj + PO)) + PF
    if np.min(RSob) <= 0.0:
      mask = RSob <= 0.0
      RSob[mask] = np.e
    else:
      mask = None

    temp_celsius = PB / np.log(RSob) - 273.15

    if mask is not None:
      min_temp = np.min(temp_celsius[np.logical_not(mask)])
      temp_celsius[mask] = min_temp

    return temp_celsius

  @staticmethod
  def extract_float(dirtystr):
    """
    Extract the float value of a string, helpful for parsing the exiftool data
    :return:
    """
    digits = re.findall(r'[-+]?\d*\.\d+|\d+', dirtystr)

    return float(digits[0])

  def save_images(self):
    """
    Save the extracted images
    :return:
    """
    rgb_np = self.get_rgb_np()
    thermal_np = self.extract_thermal_image()

    img_visual = Image.fromarray(rgb_np)
    thermal_normalized = (thermal_np - np.amin(thermal_np)) / (
      np.amax(thermal_np) - np.amin(thermal_np)
    )
    img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

    fn_prefix, _ = os.path.splitext(self.flir_img_filename)
    thermal_filename = fn_prefix + self.thermal_suffix
    image_filename = fn_prefix + self.image_suffix
    if self.use_thumbnail:
      image_filename = fn_prefix + self.thumbnail_suffix

    if self.is_debug:
      print('DEBUG Saving RGB image to:{}'.format(image_filename))
      print('DEBUG Saving Thermal image to:{}'.format(thermal_filename))

    img_visual.save(image_filename)
    img_thermal.save(thermal_filename)

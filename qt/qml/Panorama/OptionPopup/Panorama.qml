import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../../Custom"

Popup {
    id: _popup

    property int tooltip_width: 700
    property var _config: {
        "panorama": null
    }

    function combo_value(value) {
        value = value.toLowerCase();
        if (value === 'none')
            value = false;

        return value;
    }

    function reset() {
        if (!_config['panorama'])
            return ;

        let st = _config['panorama']['stitch'];
        // XXX 대소문자 문제로 {id}.model.indexOf({value}) 대신 array 직접 지정
        // 문제 발생 시, config.yaml의 옵션을 GUI 표기와 통일
        _perspective.currentIndex = ['panorama', 'scan'].indexOf(st['perspective']);
        _warp.currentIndex = ['plane', 'spherical'].indexOf(st['warp']);
        _compose_scale.value = st['compose_scale'] * 100;
        _warp_threshold.value = st['warp_threshold'] * 100;
        // -
        let bl = _config['panorama']['blend'];
        let blend_type = ['feather', 'multiband', 'no'];
        _ir_blend_type.currentIndex = blend_type.indexOf(bl['type']['IR']);
        _vis_blend_type.currentIndex = blend_type.indexOf(bl['type']['VIS']);
        _ir_blend_strength.value = bl['strength']['IR'] * 100;
        _vis_blend_strength.value = bl['strength']['VIS'] * 100;
        // -
        let prep = _config['panorama']['preprocess'];
        let contrast = ['equalization', 'normalization', null];
        let denoise = ['bilateral', 'gaussian', null];
        _ir_masking_threshold.value = prep['IR']['masking_threshold'] * 100;
        _ir_contrast.currentIndex = contrast.indexOf(prep['IR']['contrast']);
        _ir_denoise.currentIndex = denoise.indexOf(prep['IR']['denoise']);
        _vis_contrast.currentIndex = contrast.indexOf(prep['VIS']['contrast']);
        _vis_denoise.currentIndex = denoise.indexOf(prep['VIS']['denoise']);
    }

    function configure() {
        _config = {
            "panorama": {
                "stitch": {
                    "perspective": _perspective.currentText.toLowerCase(),
                    "warp": _warp.currentText.toLowerCase(),
                    "compose_scale": _compose_scale.value / 100,
                    "warp_threshold": _warp_threshold.value / 100
                },
                "blend": {
                    "type": {
                        "IR": combo_value(_ir_blend_type.currentText),
                        "VIS": combo_value(_vis_blend_type.currentText)
                    },
                    "strength": {
                        "IR": _ir_blend_strength.value / 100,
                        "VIS": _vis_blend_strength.value / 100
                    }
                },
                "preprocess": {
                    "IR": {
                        "contrast": combo_value(_ir_contrast.currentText),
                        "denoise": combo_value(_ir_denoise.currentText),
                        "masking_threshold": _ir_masking_threshold.value / 100
                    },
                    "VIS": {
                        "contrast": combo_value(_vis_contrast.currentText),
                        "denoise": combo_value(_vis_denoise.currentText)
                    }
                }
            }
        };
        con.configure(JSON.stringify(_config));
    }

    function update_config(config) {
        _config['panorama'] = config['panorama'];
        reset();
    }

    anchors.centerIn: Overlay.overlay
    Material.elevation: 5
    padding: 0
    height: _content.implicitHeight

    ColumnLayout {
        id: _content

        anchors.fill: parent

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            Layout.minimumWidth: 450
            Layout.maximumWidth: 750
            spacing: 20

            Label {
                id: _title

                Layout.fillWidth: true
                font.pointSize: 16
                font.weight: Font.Medium
                text: '파노라마 생성 설정'
            }

            ColumnLayout {
                spacing: 0

                Label {
                    Layout.fillWidth: true
                    font.weight: Font.Medium
                    font.pointSize: 13
                    text: '정합 설정'
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 20

                    Label {
                        Layout.fillWidth: true
                        text: '촬영 방법'
                    }

                    ComboBox {
                        id: _perspective

                        Layout.fillWidth: true
                        model: ['Panorama', 'Scan']

                        ToolTip {
                            text: 'Panorama(기본): 한 자리에서 카메라를 상하 좌우로 움직이며 촬영합니다.<br>Scan: 자리를 이동하며 대상에 평행하도록 촬영합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '투영 방법'
                    }

                    ComboBox {
                        id: _warp

                        Layout.fillWidth: true
                        model: ['Plane', 'Spherical']

                        ToolTip {
                            text: 'Plane(기본): 평면으로 파노라마 영상을 투영합니다.<br>Spherical: 구형으로 파노라마 영상을 투영합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '열화상 Blend 종류'
                    }

                    ComboBox {
                        id: _ir_blend_type

                        Layout.fillWidth: true
                        model: ['Feather', 'Multiband', 'None']

                        ToolTip {
                            text: '영상이 겹치는 영역이 자연스럽게 이어지게 하기 위한 처리 방법을 결정합니다.<br>Feather(기본): 겹치는 영역의 이미지를 평균하여 처리합니다.<br>Multiband: 해상도를 변경한 여러 이미지를 통해 겹치는 영역을 처리합니다.<br>None(기본): Blend를 하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '실화상 Blend 종류'
                    }

                    ComboBox {
                        id: _vis_blend_type

                        Layout.fillWidth: true
                        model: ['Feather', 'Multiband', 'None']

                        ToolTip {
                            text: '영상이 겹치는 영역이 자연스럽게 이어지게 하기 위한 처리 방법을 결정합니다.<br>Feather: 겹치는 영역의 이미지를 평균하여 처리합니다.<br>Multiband: 해상도를 변경한 여러 이미지를 통해 겹치는 영역을 처리합니다.<br>None: Blend를 하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '열화상 Blend 강도'
                    }

                    FloatSpinBox {
                        id: _ir_blend_strength

                        value: 5
                        from: 1
                        to: 100
                        stepSize: 1

                        ToolTip {
                            text: '0.05(기본). 값이 클수록 경계가 자연스럽지만 흐릿하게 보일 수 있습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '실화상 Blend 강도'
                    }

                    FloatSpinBox {
                        id: _vis_blend_strength

                        value: 5
                        from: 1
                        to: 100
                        stepSize: 1

                        ToolTip {
                            text: '0.05(기본). 값이 클수록 경계가 자연스럽지만 흐릿하게 보일 수 있습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '스케일'
                    }

                    FloatSpinBox {
                        id: _compose_scale

                        Layout.fillWidth: true
                        value: 100
                        from: 10
                        to: 100
                        stepSize: 5

                        ToolTip {
                            text: '1.00(기본). 입력한 영상 대비 생성하는 파노라마의 해상도. 영상 크기가 너무 커서 오류가 발생할 경우 낮춰주세요.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        Layout.fillWidth: true
                        text: '변형 한계'
                    }

                    FloatSpinBox {
                        id: _warp_threshold

                        Layout.fillWidth: true
                        value: 2000
                        from: 100
                        to: 10000
                        stepSize: 100
                        decimals: 1

                        ToolTip {
                            text: '20.0(기본). 파노라마 생성을 위한 각 영상의 최대 변형 정도. 클수록 가장자리의 영상도 파노라마에 포함할 수 있지만 시점 왜곡이 커지고 파노라마의 생성에 실패하거나 혹은 파일 용량이 커질 수 있습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                }

            }

            ColumnLayout {
                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 20

                    // title
                    Label {
                        Layout.columnSpan: 2
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '열화상 전처리'
                    }

                    Label {
                        Layout.columnSpan: 2
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '실화상 전처리'
                    }

                    // options
                    Label {
                        text: '명암 보정'
                    }

                    ComboBox {
                        id: _ir_contrast

                        Layout.fillWidth: true
                        model: ['Equalization', 'Normalization', 'None']

                        ToolTip {
                            text: 'Equalization(기본): 히스토그램 평활화를 통해 명암차를 극대화하고 영상 인식 정확도를 개선합니다.<br>Normalization:히스토그램 정규화를 통해 최대·최소 밝기 차이를 극대화합니다. Equalization에 비해 명암 개선 정도는 낮지만 자연스러운 영상을 얻습니다.<br>None: 명암 보정을 하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '명암 보정'
                    }

                    ComboBox {
                        id: _vis_contrast

                        Layout.fillWidth: true
                        model: ['Equalization', 'Normalization', 'None']

                        ToolTip {
                            text: 'Equalization: 히스토그램 평활화를 통해 명암차를 극대화하고 영상 인식 정확도를 개선합니다.<br>Normalization(기본):히스토그램 정규화를 통해 최대·최소 밝기 차이를 극대화합니다. Equalization에 비해 명암 개선 정도는 낮지만 자연스러운 영상을 얻습니다.<br>None: 명암 보정을 하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '노이즈 제거'
                    }

                    ComboBox {
                        id: _ir_denoise

                        Layout.fillWidth: true
                        model: ['Bilateral', 'Gaussian', 'None']

                        ToolTip {
                            text: 'Bilateral(기본): 양방향 필터를 적용하여 노이즈를 제거합니다.<br>Gaussian: 가우시안 필터를 적용하여 노이즈를 제거합니다.<br>None: 노이즈를 제거하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '노이즈 제거'
                    }

                    ComboBox {
                        id: _vis_denoise

                        Layout.fillWidth: true
                        model: ['Bilateral', 'Gaussian', 'None']

                        ToolTip {
                            text: 'Bilateral: 양방향 필터를 적용하여 노이즈를 제거합니다.<br>Gaussian(기본): 가우시안 필터를 적용하여 노이즈를 제거합니다.<br>None: 노이즈를 제거하지 않습니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                }

                RowLayout {
                    Label {
                        text: '마스킹 온도 [°C]'
                    }

                    FloatSpinBox {
                        id: _ir_masking_threshold

                        Layout.preferredWidth: _ir_denoise.width
                        value: -3000
                        from: -10000
                        to: 5000
                        stepSize: 100
                        decimals: 1

                        ToolTip {
                            text: '-30.0(기본). 설정값 미만의 영역을 하늘로 인식합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                }

            }

            RowLayout {
                Layout.alignment: Qt.AlignRight | Qt.AlignBottom

                Button {
                    flat: true
                    text: 'Cancel'
                    onClicked: {
                        reset();
                        _popup.close();
                    }
                }

                Button {
                    flat: true
                    text: 'OK'
                    onClicked: {
                        configure();
                        _popup.close();
                    }
                }

            }

        }

    }

}

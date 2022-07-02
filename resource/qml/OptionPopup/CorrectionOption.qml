import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

import '../Custom'

Popup {
    id : _popup

    property var _config: {'distort_correction':null}

    anchors.centerIn : Overlay.overlay
    Material.elevation : 5
    padding : 0
    height : _content.implicitHeight

    ColumnLayout {
        id : _content

        anchors.fill : parent

        ColumnLayout {
            Layout.fillWidth : true
            Layout.fillHeight : true
            Layout.margins : 20
            Layout.minimumWidth : 450
            Layout.maximumWidth : 750
            spacing : 20

            Label {
                id : _title
                Layout.fillWidth : true

                font.pointSize : 16
                font.weight : Font.Medium

                text : '파노라마 왜곡 보정 설정'
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '윤곽선 인식 설정'
                }

                GridLayout {
                    Layout.fillWidth : true
                    columns : 2

                    Label {
                        Layout.fillWidth : true
                        text : 'Canny 필터 표준편차'
                    }
                    FloatSpinBox {
                        id : _canny_sigma
                        Layout.fillWidth : true

                        value : 300
                        from : 10
                        to : 10000
                        stepSize : 10
                        decimals : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : 'Hough 윤곽선 검출 임계값'
                    }
                    SpinBox {
                        id : _hough_threshold
                        Layout.fillWidth : true

                        value : 10
                        from : 1
                        to : 1000
                        stepSize : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '윤곽선 최소 길이 [pixel]'
                    }
                    SpinBox {
                        id : _hough_line_length
                        Layout.fillWidth : true

                        value : 10
                        from : 1
                        to : 1000
                        stepSize : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '윤곽선 공백 허용치 [pixel]'
                    }
                    SpinBox {
                        id : _hough_line_gap
                        Layout.fillWidth : true

                        value : 5
                        from : 1
                        to : 1000
                        stepSize : 1
                    }
                }
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '왜곡 인식 설정'
                }

                GridLayout {
                    Layout.fillWidth : true
                    columns : 2

                    Label {
                        Layout.fillWidth : true
                        text : '소실점 판단 임계 각도 [º]'
                    }
                    FloatSpinBox {
                        id : _correction_threshold
                        Layout.fillWidth : true

                        value : 500
                        from : 10
                        to : 3000
                        stepSize : 50
                        decimals : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '분석 영역의 침식 강도'
                    }
                    SpinBox {
                        id : _correction_erode
                        Layout.fillWidth : true

                        value : 50
                        from : 0
                        to : 1000
                        stepSize : 10
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '최대 반복 계산 횟수'
                    }
                    SpinBox {
                        id : _correction_iteration
                        Layout.fillWidth : true

                        value : 5
                        from : 1
                        to : 10
                        stepSize : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '수직·수평 소실점만 인식'
                    }
                    CheckBox {
                        id : _correction_strict
                    }
                }
            }

            RowLayout {
                Layout.alignment : Qt.AlignRight | Qt.AlignBottom
                Button {
                    flat : true
                    text : 'Cancel'
                    onClicked : {
                        reset();
                        _popup.close()
                    }
                }

                Button {
                    flat : true
                    text : 'OK'
                    onClicked : {
                        configure();
                        _popup.close();
                    }
                }
            }
        }
    }


    function reset() {
        let cfg = _config['distort_correction']
        if (! cfg) {
            return
        }

        _canny_sigma.value = cfg['canny']['sigma'] * 100
        _hough_threshold.value = cfg['hough']['threshold']
        _hough_line_gap.value = cfg['hough']['line_gap']
        _hough_line_length.value = cfg['hough']['line_length']

        _correction_threshold.value = cfg['correction']['threshold'] * 100
        _correction_erode.value = cfg['correction']['erode']
        _correction_iteration.value = cfg['correction']['vp_iter']
        _correction_strict.checked = cfg['correction']['strict']
    }

    function configure() {
        _config = {
            'distort_correction': {
                'correction': {
                    'threshold': _correction_threshold.value / 100.0,
                    'erode': _correction_erode.value,
                    'vp_iter': _correction_iteration.value,
                    'ransac_iter': 400 * _correction_iteration.value,
                    'strict': _correction_strict.checked
                },
                'canny': {
                    'sigma': _canny_sigma.value / 100.0
                },
                'hough': {
                    'threshold': _hough_threshold.value,
                    'line_gap': _hough_line_gap.value,
                    'line_length': _hough_line_length.value
                }
            }
        }

        con.configure(JSON.stringify(_config))
    }

    function update_config(config) {
        _config['distort_correction'] = config['distort_correction']
        reset()
    }
}

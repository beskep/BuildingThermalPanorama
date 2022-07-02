import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

import '../Custom'

Popup {
    id : _popup

    property var _config: {'output':null}

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

                text : '자동 층 인식 및 저장 설정'
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '윤곽선 변환'
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
                        stepSize : 20
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

                        value : 25
                        from : 1
                        to : 1000
                        stepSize : 5
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '윤곽선 공백 허용치 [pixel]'
                    }
                    SpinBox {
                        id : _hough_line_gap
                        Layout.fillWidth : true

                        value : 10
                        from : 1
                        to : 1000
                        stepSize : 5
                    }
                }
            }

            ColumnLayout {
                spacing : 0

                Label {
                    Layout.fillWidth : true

                    font.weight : Font.Medium
                    font.pointSize : 13

                    text : '윤곽선 인식'
                }

                GridLayout {
                    Layout.fillWidth : true
                    columns : 2

                    Label {
                        Layout.fillWidth : true
                        text : '인식 대상'
                    }
                    RowLayout {
                        RadioButton {
                            id : _edgelet_seg
                            Layout.fillWidth : true
                            checked : true
                            text : '외피부위'
                        }
                        RadioButton {
                            id : _edgelet_ir
                            Layout.fillWidth : true
                            text : '열화상'
                        }
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '창문 임계치 [%]'
                    }
                    SpinBox {
                        id : _edgelet_window_threshold
                        Layout.fillWidth : true
                        enabled : _edgelet_seg.checked

                        value : 5
                        from : 0
                        to : 100
                        stepSize : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '슬라브 상대적 위치'
                    }
                    SpinBox {
                        id : _edgelet_slab_position
                        Layout.fillWidth : true
                        enabled : _edgelet_seg.checked

                        value : 50
                        from : 0
                        to : 100
                        stepSize : 5
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '최대 선정 개수'
                    }
                    SpinBox {
                        id : _edgelet_max_count
                        Layout.fillWidth : true

                        value : 10
                        from : 1
                        to : 100
                        stepSize : 1
                    }

                    Label {
                        Layout.fillWidth : true
                        text : '거리 한계 [pixel]'
                    }
                    SpinBox {
                        id : _edgelet_distance
                        Layout.fillWidth : true

                        value : 10
                        from : 1
                        to : 100
                        stepSize : 1
                    }


                    Label {
                        Layout.fillWidth : true
                        text : '각도 한계 [degree]'
                    }
                    SpinBox {
                        id : _edgelet_angle
                        Layout.fillWidth : true

                        value : 5
                        from : 1
                        to : 90
                        stepSize : 1
                    }
                }
            }

            // TODO 저장 설정

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
        let cfg = _config['output']
        if (! cfg) {
            return
        }

        _canny_sigma.value = cfg['canny']['sigma'] * 100

        _hough_threshold.value = cfg['hough']['threshold']
        _hough_line_gap.value = cfg['hough']['line_gap']
        _hough_line_length.value = cfg['hough']['line_length']

        if (cfg['edgelet']['segmentation']) {
            _edgelet_seg.checked = true;
        } else {
            _edgelet_ir.checked = true;
        }
        _edgelet_window_threshold.value = cfg['edgelet']['window_threshold'] * 100
        _edgelet_slab_position.value = cfg['edgelet']['slab_position'] * 100

        _edgelet_max_count.value = cfg['edgelet']['max_count']
        _edgelet_distance.value = cfg['edgelet']['distance_threshold']
        _edgelet_angle.value = cfg['edgelet']['angle_threshold']
    }

    function configure() {
        _config = {
            'output': {
                'canny': {
                    'sigma': _canny_sigma.value / 100.0
                },
                'hough': {
                    'threshold': _hough_threshold.value,
                    'line_gap': _hough_line_gap.value,
                    'line_length': _hough_line_length.value
                },
                'edgelet': {
                    'segmentation': _edgelet_seg.checked,
                    'window_threshold': _edgelet_window_threshold.value / 100.0,
                    'slab_position': _edgelet_slab_position.value / 100.0,
                    'max_count': _edgelet_max_count.value,
                    'distance_threshold': _edgelet_distance.value,
                    'angle_threshold': _edgelet_angle.value
                }
            }
        }

        con.configure(JSON.stringify(_config))
    }

    function update_config(config) {
        _config['output'] = config['output']
        reset()
    }
}

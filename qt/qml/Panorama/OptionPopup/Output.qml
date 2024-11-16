import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../../Custom"

Popup {
    id: _popup

    property int tooltip_width: 500
    property var _config: {
        "output": null
    }

    function reset() {
        let cfg = _config['output'];
        if (!cfg)
            return ;

        // canny
        _canny_sigma.value = cfg['canny']['sigma'] * 100;
        // hough
        _hough_threshold.value = cfg['hough']['threshold'];
        _hough_line_gap.value = cfg['hough']['line_gap'];
        _hough_line_length.value = cfg['hough']['line_length'];
        // edgelet
        if (cfg['edgelet']['segmentation'])
            _edgelet_seg.checked = true;
        else
            _edgelet_ir.checked = true;
        _edgelet_window_threshold.value = cfg['edgelet']['window_threshold'] * 100;
        _edgelet_slab_position.value = cfg['edgelet']['slab_position'] * 100;
        _edgelet_max_count.value = cfg['edgelet']['max_count'];
        _edgelet_distance.value = cfg['edgelet']['distance_threshold'];
        _edgelet_angle.value = cfg['edgelet']['angle_threshold'];
        // segment
        _split_count.checked = cfg['segment']['method'] === 'count';
        _segments_count.value = cfg['segment']['count'];
        _segments_length.text = cfg['segment']['length'];
        _building_width.text = cfg['segment']['building_width'];
    }

    function configure() {
        _config = {
            "output": {
                "canny": {
                    "sigma": _canny_sigma.value / 100
                },
                "hough": {
                    "threshold": _hough_threshold.value,
                    "line_gap": _hough_line_gap.value,
                    "line_length": _hough_line_length.value
                },
                "edgelet": {
                    "segmentation": _edgelet_seg.checked,
                    "window_threshold": _edgelet_window_threshold.value / 100,
                    "slab_position": _edgelet_slab_position.value / 100,
                    "max_count": _edgelet_max_count.value,
                    "distance_threshold": _edgelet_distance.value,
                    "angle_threshold": _edgelet_angle.value
                },
                "segment": {
                    "method": _split_count.checked ? "count" : "length",
                    "count": _segments_count.value,
                    "length": parseFloat(_segments_length.text),
                    "building_width": parseFloat(_building_width.text)
                }
            }
        };
        con.configure(JSON.stringify(_config));
    }

    function update_config(config) {
        _config['output'] = config['output'];
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
            Layout.minimumWidth: 500
            Layout.maximumWidth: 1000
            spacing: 20

            Label {
                id: _title

                Layout.fillWidth: true
                font.pointSize: 16
                font.weight: Font.Medium
                text: '자동 층 인식 설정'
            }

            RowLayout {
                spacing: 50

                ColumnLayout {
                    Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                    spacing: 0

                    Label {
                        Layout.fillWidth: true
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '윤곽선 변환'
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2

                        Label {
                            Layout.fillWidth: true
                            text: 'Canny 필터 표준편차'
                        }

                        FloatSpinBox {
                            id: _canny_sigma

                            Layout.fillWidth: true
                            value: 300
                            from: 10
                            to: 10000
                            stepSize: 20
                            decimals: 1

                            ToolTip {
                                text: '윤곽선 검출을 위한 가우시안 필터의 표준편차입니다. 클수록 민감도가 낮아지지만 노이즈의 영향이 감소합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: 'Hough 윤곽선 검출 임계값'
                        }

                        SpinBox {
                            id: _hough_threshold

                            Layout.fillWidth: true
                            value: 10
                            from: 1
                            to: 1000
                            stepSize: 1

                            ToolTip {
                                text: '직선 윤곽선 검출을 위한 임계값입니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '윤곽선 최소 길이 [pixel]'
                        }

                        SpinBox {
                            id: _hough_line_length

                            Layout.fillWidth: true
                            value: 25
                            from: 1
                            to: 1000
                            stepSize: 5

                            ToolTip {
                                text: '설정치보다 긴 윤곽선만 검출됩니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '윤곽선 공백 허용치 [pixel]'
                        }

                        SpinBox {
                            id: _hough_line_gap

                            Layout.fillWidth: true
                            value: 10
                            from: 1
                            to: 1000
                            stepSize: 5

                            ToolTip {
                                text: '직선 중간에 설정치보다 작은 공백이 존재해도 윤곽선으로 인식합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                    }

                    Rectangle {
                        height: 20
                    }

                    Label {
                        Layout.fillWidth: true
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '층 인식'
                    }

                    RowLayout {
                        spacing: 25

                        RadioButton {
                            id: _split_count

                            Layout.fillWidth: true
                            text: '분할 개수 설정'
                            checked: true

                            ToolTip {
                                text: '건물의 폭을 지정한 개수로 분할합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        RadioButton {
                            id: _split_length

                            Layout.fillWidth: true
                            text: '분할 길이 설정'

                            ToolTip {
                                text: '건물의 폭을 일정한 길이로 분할합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 25

                        RowLayout {
                            enabled: _split_count.checked

                            Label {
                                text: '분할 개수'
                            }

                            SpinBox {
                                id: _segments_count

                                Layout.fillWidth: true
                                wheelEnabled: true
                                value: 20

                                ToolTip {
                                    text: '지정한 개수만큼 건물의 폭을 분할합니다.'
                                    implicitWidth: tooltip_width
                                    visible: parent.hovered
                                }

                            }

                        }

                        RowLayout {
                            enabled: _split_length.checked

                            Label {
                                text: '분할 길이'
                            }

                            TextField {
                                id: _segments_length

                                Layout.fillWidth: true
                                horizontalAlignment: TextInput.AlignRight
                                text: '0.05'

                                ToolTip {
                                    text: '건물 폭을 분할하는 길이를 설정합니다.'
                                    implicitWidth: tooltip_width
                                    visible: parent.hovered
                                }

                                validator: DoubleValidator {
                                }

                            }

                            Label {
                                text: 'm'
                            }

                        }

                        Label {
                        }

                        RowLayout {
                            enabled: _split_length.checked

                            Label {
                                text: '건물 폭'
                            }

                            TextField {
                                id: _building_width

                                Layout.fillWidth: true
                                horizontalAlignment: TextInput.AlignRight
                                text: ''

                                ToolTip {
                                    text: '건물의 폭을 설정합니다.'
                                    implicitWidth: tooltip_width
                                    visible: parent.hovered
                                }

                                validator: DoubleValidator {
                                }

                            }

                            Label {
                                text: 'm'
                            }

                        }

                    }

                }

                ColumnLayout {
                    Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                    spacing: 0

                    Label {
                        Layout.fillWidth: true
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '윤곽선 인식'
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2

                        Label {
                            Layout.fillWidth: true
                            text: '인식 대상'
                        }

                        RowLayout {
                            RadioButton {
                                id: _edgelet_seg

                                Layout.fillWidth: true
                                checked: true
                                text: '외피부위'
                            }

                            RadioButton {
                                id: _edgelet_ir

                                Layout.fillWidth: true
                                text: '열화상'
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '창문 임계치 [%]'
                        }

                        SpinBox {
                            id: _edgelet_window_threshold

                            Layout.fillWidth: true
                            wheelEnabled: true
                            enabled: _edgelet_seg.checked
                            value: 50
                            from: 0
                            to: 100
                            stepSize: 5

                            ToolTip {
                                text: '층 인식 결과 중 창문을 제외하기 위한 임계치입니다. 외피 영역 중 임계치보다 창문 영역의 비율이 높으면 층 구분선이 아니라고 판단합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '슬라브 상대 위치 [%]'
                        }

                        SpinBox {
                            id: _edgelet_slab_position

                            Layout.fillWidth: true
                            wheelEnabled: true
                            enabled: _edgelet_seg.checked
                            value: 50
                            from: 0
                            to: 100
                            stepSize: 5

                            ToolTip {
                                text: '층 구분선 사이 영역 중 슬라브가 위치하는 높이를 설정합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '최대 선정 개수'
                        }

                        SpinBox {
                            id: _edgelet_max_count

                            Layout.fillWidth: true
                            wheelEnabled: true
                            value: 10
                            from: 1
                            to: 100
                            stepSize: 1

                            ToolTip {
                                text: '층의 최대 인식 개수를 설정합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '거리 한계 [pixel]'
                        }

                        SpinBox {
                            id: _edgelet_distance

                            Layout.fillWidth: true
                            wheelEnabled: true
                            value: 10
                            from: 1
                            to: 100
                            stepSize: 1

                            ToolTip {
                                text: '지정한 거리 이상 떨어진 층만 인식합니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

                        }

                        Label {
                            Layout.fillWidth: true
                            text: '각도 한계 [°]'
                        }

                        SpinBox {
                            id: _edgelet_angle

                            Layout.fillWidth: true
                            wheelEnabled: true
                            value: 5
                            from: 1
                            to: 90
                            stepSize: 1

                            ToolTip {
                                text: '수평과 설정 각도 이상 차이가 존재하면 층 구분선으로 인식하지 않습니다.'
                                implicitWidth: tooltip_width
                                visible: parent.hovered
                            }

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

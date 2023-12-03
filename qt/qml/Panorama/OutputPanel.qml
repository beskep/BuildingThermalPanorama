import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.qmlmodels 1.0
import "../Custom"
import "OptionPopup"
import Backend 1.0

Pane {
    property int mode_height: 36

    function output_plot() {
        let image = '';
        if (_image_ir.checked)
            image = 'ir';
        else if (_image_edges.checked)
            image = 'edges';
        else if (_image_vis.checked)
            image = 'vis';
        else
            image = 'seg';
        con.output_plot(image);
    }

    function init() {
        _option.configure();
        output_plot();
    }

    function update_config(config) {
        _option.update_config(config);
    }

    function save_output() {
        con.output_save(_split_count.checked, _segments_count.value, parseFloat(_segments_length.text), parseFloat(_building_width.text));
    }

    width: 1280
    height: 720
    padding: 10
    objectName: 'output_panel'

    OutputOption {
        id: _option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            spacing: 0

            RowLayout {
                ToolButton {
                    text: '자동 추정'
                    icon: '\ue663'
                    onReleased: con.output_estimate_edgelets()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('층 구분선 자동 추정')
                }

                ToolButton {
                    id: _delete

                    text: '전체 삭제'
                    icon: '\ue872'
                    onReleased: con.output_clear_lines()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('층 구분선 전체 삭제')
                }

                ToolButton {
                    text: '저장'
                    icon: '\ue161'
                    onReleased: save_output()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('GIS 연동을 위한 층별 온도 데이터 저장')
                }

                ToolButton {
                    text: '설정'
                    icon: '\ue8b8'
                    onReleased: _option.open()
                }

                ToolButton {
                    text: '도움말'
                    icon: '\ue88e'
                    ToolTip.visible: hovered
                    ToolTip.text: '자동 추정: 영상 윤곽선으로부터 층 구분선 추정\n마우스 좌클릭: 구분선 추가 또는 수정\n마우스 우클릭: 구분선 삭제'
                }

            }

        }

        Pane {
            Material.elevation: 2
            Layout.fillHeight: true
            Layout.fillWidth: true
            padding: 0

            FigureCanvas {
                id: plot

                anchors.fill: parent
                objectName: 'output_plot'
                dpi_ratio: Screen.devicePixelRatio
            }

            // plot 종류 선택
            Pane {
                Material.elevation: 1
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                padding: 0

                RowLayout {
                    RowLayout {
                        visible: _expand_button.expanded

                        RadioButton {
                            id: _image_ir

                            Layout.preferredHeight: mode_height
                            text: '열화상'
                            checked: true
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_edges

                            Layout.preferredHeight: mode_height
                            text: '윤곽선'
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_vis

                            Layout.preferredHeight: mode_height
                            text: '실화상'
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_seg

                            Layout.preferredHeight: mode_height
                            text: '외피 부위'
                            onReleased: output_plot()
                        }

                        ToolSeparator {
                            color: '#9e9e9e'
                        }

                        CheckBox {
                            Layout.preferredHeight: mode_height
                            text: '선분 연장'
                            onCheckedChanged: con.output_extend_lines(checked)
                            ToolTip.visible: hovered
                            ToolTip.delay: 500
                            ToolTip.text: qsTr('영상 전체 범위로 구분선 자동 연장')
                        }

                    }

                    ExpandButton {
                        id: _expand_button
                    }

                }

            }

        }

        // 하단 옵션 패널
        Pane {
            Material.elevation: 2
            Layout.fillWidth: true

            ColumnLayout {
                ButtonGroup {
                    id: _split_group
                }

                RowLayout {
                    spacing: 25

                    ColumnLayout {
                        RadioButton {
                            id: _split_count

                            ButtonGroup.group: _split_group
                            font.weight: Font.Medium
                            text: '분할 개수 설정'
                            checked: true
                        }

                        RowLayout {
                            enabled: _split_count.checked

                            Label {
                                text: '분할 개수'
                            }

                            SpinBox {
                                id: _segments_count

                                value: 20
                            }

                        }

                    }

                    ColumnLayout {
                        RadioButton {
                            id: _split_length

                            ButtonGroup.group: _split_group
                            font.weight: Font.Medium
                            text: '분할 길이 설정'
                        }

                        RowLayout {
                            spacing: 25
                            enabled: _split_length.checked

                            RowLayout {
                                Label {
                                    text: '분할 길이'
                                }

                                TextField {
                                    id: _segments_length

                                    horizontalAlignment: TextInput.AlignRight
                                    text: '0.05'

                                    validator: DoubleValidator {
                                    }

                                }

                                Label {
                                    text: 'm'
                                }

                            }

                            RowLayout {
                                Label {
                                    text: '건물 폭'
                                }

                                TextField {
                                    id: _building_width

                                    horizontalAlignment: TextInput.AlignRight
                                    text: ''

                                    validator: DoubleValidator {
                                    }

                                }

                                Label {
                                    text: 'm'
                                }

                            }

                        }

                    }

                }

            }

        }

    }

}

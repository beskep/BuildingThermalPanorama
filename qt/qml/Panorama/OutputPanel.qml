import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.qmlmodels 1.0
import "../Custom"
import "../Button" as Btn
import "OptionPopup" as Opt
import Backend 1.0

Pane {
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
        con.output_save();
    }

    width: 1280
    height: 720
    padding: 10
    objectName: 'output_panel'

    Opt.Output {
        id: _option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            spacing: 0

            RowLayout {
                Btn.ToolButton {
                    text: '자동 추정'
                    icon: '\ue663'
                    onReleased: con.output_estimate_edgelets()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('층 구분선 자동 추정')
                }

                Btn.ToolButton {
                    text: '저장'
                    icon: '\ue161'
                    onReleased: save_output()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('GIS 연동을 위한 층별 온도 데이터 저장')
                }

                ToolSeparator {
                }

                Btn.OpenFolder {
                    onReleased: con.open_dir('OUT')
                }

                ToolSeparator {
                }

                Btn.Navigation {
                    index: 7
                }

                ToolSeparator {
                }

                Btn.Setting {
                    onReleased: _option.open()
                }

                Btn.Help {
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

            // 툴바
            Pane {
                anchors.left: parent.left
                anchors.top: parent.top
                Material.elevation: 1
                padding: 0
                leftPadding: 5

                RowLayout {
                    RowLayout {
                        visible: _expand.expanded

                        Btn.MiniToolButton {
                            text: '초기화'
                            icon: '\uf053'
                            ToolTip.text: '층 구분선 전체 삭제'
                            onReleased: con.output_clear_lines()
                        }

                        ToolSeparator {
                            leftPadding: 2
                            rightPadding: 2
                        }

                        RadioButton {
                            id: _image_ir

                            implicitHeight: 36
                            text: '열화상'
                            checked: true
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_edges

                            implicitHeight: 36
                            text: '윤곽선'
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_vis

                            implicitHeight: 36
                            text: '실화상'
                            onReleased: output_plot()
                        }

                        RadioButton {
                            id: _image_seg

                            implicitHeight: 36
                            text: '외피 부위'
                            onReleased: output_plot()
                        }

                        ToolSeparator {
                            leftPadding: 2
                            rightPadding: 2
                        }

                        CheckBox {
                            implicitHeight: 36
                            text: '선분 연장'
                            onCheckedChanged: con.output_extend_lines(checked)
                            ToolTip.visible: hovered
                            ToolTip.delay: 500
                            ToolTip.text: qsTr('영상 전체 범위로 구분선 자동 연장')
                        }

                    }

                    Btn.Expand {
                        id: _expand

                        padding: 0
                    }

                }

            }

        }

    }

}

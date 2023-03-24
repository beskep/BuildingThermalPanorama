import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import Qt.labs.platform 1.1

import '../Custom'
import Backend 1.0


Pane {
    property var mode: 0 // [analysis, anomaly, report]

    width : 1280
    height : 720
    padding : 10

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 0

                RowLayout {
                    visible : mode === 0

                    ToolButton {
                        text : '경로 선택'
                    }

                    ToolButton {
                        text : '영상 변환'
                    }
                }

                RowLayout {
                    visible : mode === 1

                    ToolButton {
                        text : '이상 영역 검출'
                    }
                }

                RowLayout {
                    visible : mode === 2

                    ToolButton {
                        text : '보고서 생성'
                    }
                    ToolButton {
                        text : '저장'
                    }
                }
            }
        }

        RowLayout {
            Layout.fillHeight : true
            Layout.fillWidth : true
            spacing : 10

            // 영상 목록
            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.preferredWidth : 300
                padding : 5

                ListView {
                    id : image_view

                    anchors.fill : parent
                    clip : true

                    ScrollBar.vertical : ScrollBar {
                        policy : ScrollBar.AsNeeded
                    }

                    model : ListModel {
                        id : image_model
                    }

                    delegate : Pane {
                        Material.elevation : 0
                        width : image_view.width - 20
                        height : width * 3 / 4 + 10

                        Image {
                            id : _image
                            source : path
                            width : parent.width
                            fillMode : Image.PreserveAspectFit
                        }

                        BrightnessContrast {
                            id : _bc
                            anchors.fill : _image
                            source : _image
                            brightness : 0
                        }

                        MouseArea {
                            anchors.fill : parent
                            hoverEnabled : true

                            onReleased : con.rgst_plot(path)
                            onEntered : _bc.brightness = -0.25
                            onExited : _bc.brightness = 0
                        }
                    }
                }
            }

            // plot
            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true
                padding : 0

                FigureCanvas {
                    id : _plot
                    anchors.fill : parent

                    objectName : 'plot'
                    Layout.fillHeight : true
                    Layout.fillWidth : true
                    dpi_ratio : Screen.devicePixelRatio
                }
            }
        }
    }
}

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0

import 'Custom'
import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'registration_panel'

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 10

                ToolButton {
                    text : qsTr('자동 정합')
                    icon : '\ue663'
                    onReleased : con.command('register')
                }
                ToolButton {
                    text : qsTr('저장')
                    icon : '\ue161'
                    onReleased : con.rgst_save()
                    // TODO 작업 진행 현황 따라 색 변환
                }
                ToolButton {
                    text : qsTr('취소')
                    icon : '\ue14a'
                    // TODO
                }
            }
        }

        RowLayout {
            spacing : 10

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

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true
                padding : 0

                FigureCanvas {
                    id : plot
                    anchors.fill : parent

                    objectName : 'registration_plot'
                    Layout.fillHeight : true
                    Layout.fillWidth : true
                    dpi_ratio : Screen.devicePixelRatio
                }
            }
        }
    }

    function init() {}

    function update_image_view(paths) {
        image_model.clear()
        paths.forEach(path => image_model.append({'path': path}))
    }
}

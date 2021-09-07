import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'registration_panel'

    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Button {
                text : qsTr('자동 정합')
                onReleased : con.command('register')
                // TODO 개별 자동 정합 기능 추가?
            }
            Button {
                text : qsTr('수동 정합')
            }
            Rectangle {
                width : 10
            }
            Button {
                text : qsTr('저장')
                // TODO 작업 진행 현황 따라 색 변환
            }
            Button {
                text : qsTr('취소')
            }
        }

        RowLayout {
            spacing : 10

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.preferredWidth : 300

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
                            source : path
                            width : parent.width
                            fillMode : Image.PreserveAspectFit
                        }

                        MouseArea {
                            anchors.fill : parent
                            hoverEnabled : true

                            onReleased : con.rgst_plot(path)
                        }
                    }
                }
            }

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true

                ColumnLayout {
                    anchors.fill : parent

                    RegistrationCanvas {
                        id : plot
                        objectName : 'registration_plot'
                        Layout.fillHeight : true
                        Layout.fillWidth : true
                        dpi_ratio : Screen.devicePixelRatio
                    }
                }
            }
        }
    }

    function update_image_view(paths) {
        image_model.clear()
        paths.forEach(path => image_model.append({'path': path}))
    }
}

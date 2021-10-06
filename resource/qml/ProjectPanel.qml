import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1

import 'Custom'


Pane {
    width : 1280
    height : 720
    padding : 10

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 0

                ToolButton {
                    text : qsTr('경로 선택')
                    icon : '\ue2c7'
                    onReleased : folder_dialog.open()
                }
                ToolButton {
                    text : qsTr('프로젝트 초기화')
                    icon : '\ue97a'
                    onReleased : con.prj_init_directory()
                }
                ToolButton {
                    text : qsTr('열화상 데이터 추출')
                    icon : '\ue8a7'
                    onReleased : con.command('extract')
                }
            }
        }

        RowLayout {
            Layout.fillHeight : true
            Layout.fillWidth : true
            spacing : 10

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.preferredWidth : 300

                ScrollView {
                    anchors.fill : parent
                    clip : true

                    ScrollBar.vertical.policy : ScrollBar.AsNeeded
                    ScrollBar.horizontal.policy : ScrollBar.AlwaysOff

                    Text {
                        id : project_tree
                        anchors.fill : parent
                        font.family : 'Fira Code'
                        // TODO kr font
                    }
                }
            }

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true
                padding : 10

                GridView {
                    id : image_view

                    anchors.fill : parent
                    clip : true
                    cellWidth : width / Math.ceil(width / 300)
                    cellHeight : cellWidth * 3 / 4 + 20

                    ScrollBar.vertical : ScrollBar {
                        policy : ScrollBar.AsNeeded
                    }

                    model : ListModel {
                        id : grid_model
                    }

                    delegate : Pane {
                        Material.elevation : 0
                        width : image_view.cellWidth
                        height : image_view.cellHeight

                        Column {
                            anchors.fill : parent
                            anchors.horizontalCenter : parent.horizontalCenter
                            spacing : 5

                            Image {
                                source : path
                                anchors.horizontalCenter : parent.horizontalCenter
                                width : parent.width
                                fillMode : Image.PreserveAspectFit
                            }
                            Text {
                                text : file_name(path)
                                font.family : 'Fira Code'
                                anchors.horizontalCenter : parent.horizontalCenter
                            }
                        }
                    }
                }
            }
        }
    }

    FolderDialog {
        id : folder_dialog

        onAccepted : {
            var path = folder.toString().replace('file:///', '');
            con.prj_select_working_dir(path);
        }
    }

    function init() {}

    function update_project_tree(text) {
        project_tree.text = text
    }

    function file_name(path) {
        var parts = path.split('/')

        return parts[parts.length - 1]
    }

    function update_image_view(paths) {
        grid_model.clear()
        paths.forEach(path => grid_model.append({'path': path}))
    }
}

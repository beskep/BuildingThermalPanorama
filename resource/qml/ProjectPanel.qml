import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1


Pane {
    width : 1280
    height : 720
    padding : 10

    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Layout.fillWidth : true

            Button {
                text : qsTr('경로 선택')

                onReleased : folder_dialog.open()
            }
            Button {
                text : qsTr('프로젝트 초기화')

                onReleased : {
                    con.init_directory();
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
                Layout.preferredWidth : 450
                // TODO 적정 width 설정

                ScrollView {
                    anchors.fill : parent

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
                    id : image_grid

                    anchors.fill : parent
                    clip : true
                    cellWidth : 220
                    cellHeight : 190
                    // TODO cellWidth Pane width에 따라 설정

                    ScrollBar.vertical : ScrollBar {
                        policy : ScrollBar.AsNeeded
                    }

                    model : ListModel {
                        id : grid_model
                    }

                    delegate : Pane {
                        Material.elevation : 0
                        width : image_grid.cellWidth - 20

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
            con.select_working_dir(path);
        }
    }

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

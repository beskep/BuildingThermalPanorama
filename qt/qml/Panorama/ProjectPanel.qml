import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.platform 1.1
import "../Custom"
import "OptionPopup"

Pane {
    property bool has_working_dir: false
    property string _warning: ' <u>설정 변경 시 부위 인식 및 파노라마 생성 결과가 초기화됩니다.</u>'

    function init() {
        con.has_working_dir();
    }

    function update_project_tree(text) {
        project_tree.text = text;
    }

    function file_name(path) {
        var parts = path.split('/');
        return parts[parts.length - 1];
    }

    function update_image_view(paths) {
        grid_model.clear();
        paths.forEach((path) => grid_model.append({
            "path": path
        }));
    }

    width: 1280
    height: 720
    padding: 10

    ProjectOption {
        id: _option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                spacing: 0

                ToolButton {
                    text: '프로젝트 폴더 선택'
                    icon: '\ue8a7'
                    font.pointSize: 13
                    onReleased: folder_dialog.open()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: '열화상 파노라마가 저장된 작업 경로 선택'
                }

                ToolButton {
                    text: '열화상 추출·저장'
                    icon: '\ue161'
                    font.pointSize: 13
                    enabled: has_working_dir
                    onReleased: con.command('extract')
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: 'Raw 파일로부터 열화상과 실화상 데이터 추출'
                }

                ToolSeparator {
                }

                ToolButton {
                    text: '폴더 열기'
                    icon: '\ue2c8'
                    onReleased: con.open_dir('IR')
                }

                ToolSeparator {
                }

                ToolSeparator {
                }

                ToolButton {
                    text: '이전'
                    icon: '\ueac3'
                    enabled: false
                }

                ToolButton {
                    text: '다음'
                    icon2: '\ueac9'
                    onReleased: app.set_panel(1)
                }

                ToolSeparator {
                }

                ToolSeparator {
                }

                ToolButton {
                    text: '설정' // TODO test
                    text_color: '#A0FFFFFF'
                    icon: '\ue8b8'
                    onReleased: _option.open()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: '프로젝트 설정'
                }

                ToolButton {
                    text: '도움말' // TODO
                    text_color: '#A0FFFFFF'
                    icon: '\ue88e'
                }

                ToolSeparator {
                }

            }

        }

        RowLayout {
            Layout.fillHeight: true
            Layout.fillWidth: true
            spacing: 10

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.preferredWidth: 300

                ScrollView {
                    anchors.fill: parent
                    clip: true
                    ScrollBar.vertical.policy: ScrollBar.AsNeeded
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    Text {
                        id: project_tree

                        anchors.fill: parent
                        font.family: 'Iosevka SS11'
                        font.pointSize: 10
                    }

                }

            }

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.fillWidth: true
                padding: 10

                GridView {
                    id: image_view

                    anchors.fill: parent
                    clip: true
                    cellWidth: width / Math.ceil(width / 300)
                    cellHeight: cellWidth * 3 / 4 + 20

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    model: ListModel {
                        id: grid_model
                    }

                    delegate: Pane {
                        Material.elevation: 0
                        width: image_view.cellWidth
                        height: image_view.cellHeight

                        Column {
                            anchors.fill: parent
                            anchors.horizontalCenter: parent.horizontalCenter
                            spacing: 5

                            Image {
                                source: path
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: parent.width
                                fillMode: Image.PreserveAspectFit
                            }

                            Text {
                                text: file_name(path)
                                font.family: 'Iosevka SS11'
                                font.pointSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                            }

                        }

                    }

                }

            }

        }

    }

    FolderDialog {
        id: folder_dialog

        onAccepted: {
            var path = folder.toString().replace('file:///', '');
            con.prj_select_working_dir(path);
            con.has_working_dir();
        }
    }

}

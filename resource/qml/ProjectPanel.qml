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

    property bool path_selected : false
    property string _warning : ' <u>설정 변경 시 부위 인식 및 파노라마 생성 결과가 초기화됩니다.</u>'

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 0

                ToolButton {
                    text : qsTr('경로 선택')
                    icon : '\ue8a7'
                    onReleased : folder_dialog.open()

                    ToolTip.visible : hovered
                    ToolTip.delay : 500
                    ToolTip.text : qsTr('열화상 파노라마가 저장된 작업 경로 선택')
                }
                // TODO 옵션 전체 초기화 버튼
                ToolButton {
                    text : qsTr('열화상 추출')
                    icon : '\ue30d'
                    onReleased : con.command('extract')

                    ToolTip.visible : hovered
                    ToolTip.delay : 500
                    ToolTip.text : qsTr('Raw 파일로부터 열화상과 실화상 데이터 추출')
                }

                ToolSeparator {}

                RowLayout {
                    enabled : path_selected

                    Label {
                        text : '열·실화상 파노라마 생성 설정'
                        font.weight : Font.Medium
                    }
                    ToolRadioButton {
                        checked : true
                        text : '동시 생성'

                        onReleased : set_separate_panorama()

                        ToolTip.visible : hovered
                        ToolTip.text : '개별 열·실화상을 정합 후 파노라마를 동시에 생성합니다.' + _warning
                    }
                    ToolRadioButton {
                        id : _separate

                        text : '별도 생성'

                        onReleased : set_separate_panorama()

                        ToolTip.visible : hovered
                        ToolTip.text : '열·실화상 파노라마를 별도로 생성하고 두 파노라마를 정합합니다.' + _warning
                    }
                }

                ToolSeparator {}

                ToolButton {
                    text : qsTr('실화상 입력')
                    icon : '\ue439'

                    enabled : _separate.checked

                    // onReleased : // TODO

                    ToolTip.visible : hovered
                    ToolTip.delay : 500
                    ToolTip.text : qsTr('[옵션] 열화상과 함께 촬영된 실화상이 아닌 별도의 실화상 입력')
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
            path_selected = true;
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

    function set_separate_panorama() {
        let config = {
            'panorama': {
                'separate': _separate.checked
            }
        }
        con.configure(JSON.stringify(config));
        app.separate_panorama = _separate.checked;
        app.popup('정합 설정 변경', '부위 인식 및 파노라마 생성 결과가 초기화됐습니다.', 5000)
    }

    function update_image_view(paths) {
        grid_model.clear()
        paths.forEach(path => grid_model.append({'path': path}))
    }

    function update_config(config) {
        let separate = config['panorama']['separate']
        _separate.checked = separate
        app.separate_panorama = separate
    }
}

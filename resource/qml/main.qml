import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15


ApplicationWindow {
    id : root

    property alias project_panel : project_panel

    width : 1600
    height : 900
    visible : true
    title : qsTr('건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램')

    FontLoader {
        source : '../font/FiraCode-Regular.ttf'
    }
    FontLoader {
        source : '../font/NotoSansCJKkr-DemiLight.otf'
    }

    RowLayout {
        anchors.fill : parent

        VertTabBar {
            id : tab_bar

            Layout.preferredWidth : 200
            Layout.fillHeight : true

            background : Rectangle {}

            TabButton {
                text : '프로젝트 설정'
                width : parent.width
            }
            TabButton {
                text : '열·실화상 정합'
                width : parent.width
            }
            TabButton {
                text : '외피 부위 인식'
                width : parent.width
            }
            TabButton {
                text : '파노라마 생성'
                width : parent.width
            }
            TabButton {
                text : '왜곡 보정'
                width : parent.width
            }
        }

        Page {
            Layout.fillHeight : true
            Layout.fillWidth : true

            StackLayout {
                currentIndex : tab_bar.currentIndex
                anchors.fill : parent

                ProjectPanel {
                    id : project_panel
                }

                Page {
                    Label {
                        text : '열·실화상 정합'
                    }
                }
                Page {
                    Label {
                        text : '외피 부위 인식'
                    }
                }
                Page {
                    Label {
                        text : '파노라마 생성'
                    }
                }
                Page {
                    Label {
                        text : '왜곡 보정'
                    }
                }
            }
        }
    }

    footer : StatusBar {}

    function update_project_tree(text) {
        project_panel.update_project_tree(text);
    }

    function update_image_view(paths) {
        project_panel.update_image_view(paths)
    }
}

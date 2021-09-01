import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12
import QtQuick.Window 2.12


ApplicationWindow {
    id : root

    width : 1600
    height : 900
    visible : true
    title : qsTr('건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램')

    FontLoader {
        id: mono
        source : '../font/FiraCode-Regular.ttf'
    }
    FontLoader {
        id: sans
        source : '../font/SpoqaHanSansNeo-Regular.ttf'
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
                RegistrationPanel {
                    id : registration_panel
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

    function get_panel(name) {
        if (name == 'project') {
            return project_panel
        } else if (name == 'registration') {
            return registration_panel
        }

        return null
    }
}

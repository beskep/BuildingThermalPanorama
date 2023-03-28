import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import 'Custom'
import 'EGS'


ApplicationWindow {
    id : app
    property ApplicationWindow app : app
    property string resource : '../../resource'

    width : 1600
    height : 900
    visible : true
    title : '외피 열적 이상 영역 자동 검출 솔루션'

    FontLoader {
        id : mono
        source : `${resource}/font/FiraCode-Regular.ttf`
    }
    FontLoader {
        id : sans
        source : `${resource}/font/SourceHanSansKR-Normal.otf`
    }
    FontLoader {
        source : `${resource}/font/SourceHanSansKR-Medium.otf`
    }
    FontLoader {
        id : icon
        source : `${resource}/font/MaterialIcons-Regular.ttf`
    }

    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Layout.fillWidth : true
            Layout.fillHeight : true

            VertTabBar {
                id : tab_bar

                Layout.preferredWidth : 200
                Layout.fillHeight : true

                background : Rectangle {}

                TabButton {
                    text : '열화상 입력'
                    width : parent.width
                }
                TabButton {
                    text : '열적 이상 영역 검출'
                    width : parent.width
                }
                TabButton {
                    text : '보고서 출력'
                    width : parent.width
                }

                onCurrentIndexChanged : panel.mode = currentIndex
            }

            MainPanel {
                id : panel

                Layout.fillHeight : true
                Layout.fillWidth : true
            }
        }

        ProgressBar {
            id : _pb
            Layout.fillWidth : true
            indeterminate : false
            value : 1.0
        }
    }

    Image {
        source : `${resource}/EGSolutionsLogoKR.svg`
        sourceSize.width : tab_bar.width - 10

        anchors.left : parent.left
        anchors.leftMargin : 5

        anchors.bottom : parent.bottom
        anchors.bottomMargin : 10
    }

    TimeoutPopup {
        id : _popup
    }

    footer : StatusBar {
        id : _status_bar
        text : '외피 열적 이상 영역 자동 검출 솔루션'
    }

    function pb_value(value) {
        _pb.value = value
    }

    function pb_state(indeterminate) {
        _pb.indeterminate = indeterminate
    }

    function status_message(msg) {
        _status_bar.text = msg
    }

    function popup(title, message, timeout = 2000) {
        _popup.timeout_open(title, message, timeout)
    }

    function get_panel() {
        return panel
    }
}

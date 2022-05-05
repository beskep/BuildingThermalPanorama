import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import 'Custom'


ApplicationWindow {
    id : app
    property ApplicationWindow app : app
    property bool separate_panorama : false

    width : 1600
    height : 900
    visible : true
    title : qsTr('건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램')

    FontLoader {
        id : mono
        source : '../font/FiraCode-Regular.ttf'
    }
    FontLoader {
        id : sans
        source : '../font/NotoSansCJKkr-DemiLight.otf'
    }
    FontLoader {
        source : '../font/NotoSansCJKkr-Medium.otf'
    }
    FontLoader {
        id : icon
        source : '../font/MaterialIcons-Regular.ttf'
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
                    text : '프로젝트 설정'
                    width : parent.width
                }
                TabButton {
                    text : '열·실화상 정합'
                    width : parent.width

                    enabled : !separate_panorama
                }
                TabButton {
                    text : '외피 부위 인식'
                    width : parent.width
                }
                TabButton {
                    text : '파노라마 생성·보정'
                    width : parent.width
                }
                TabButton {
                    text : '파노라마 정합'
                    width : parent.width

                    enabled : separate_panorama
                }
                TabButton {
                    text : '온도 분포'
                    width : parent.width
                }
            }

            Page {
                Layout.fillHeight : true
                Layout.fillWidth : true

                StackLayout {
                    anchors.fill : parent

                    // 파노라마 정합 선택 시 registration_panel (1번) 표시
                    currentIndex : [
                        0,
                        1,
                        2,
                        3,
                        1,
                        4
                    ][tab_bar.currentIndex]

                    onCurrentIndexChanged : {
                        [
                            project_panel,
                            registration_panel,
                            segmentation_panel,
                            panorama_panel,
                            registration_panel,
                            descriptive_panel
                        ][currentIndex].init()
                    }

                    ProjectPanel {
                        id : project_panel
                    }
                    RegistrationPanel {
                        id : registration_panel
                        // 파노라마 정합 선택 시
                        separate_panorama : separate_panorama
                    }
                    SegmentationPanel {
                        id : segmentation_panel
                    }
                    PanoramaPanel {
                        id : panorama_panel
                    }
                    DescriptivePanel {
                        id : descriptive_panel
                    }
                }
            }
        }

        ProgressBar {
            id : _pb
            Layout.fillWidth : true
            indeterminate : false
            value : 1.0
        }
    }

    TimeoutPopup {
        id : _popup
    }

    footer : StatusBar {
        id : _status_bar
    }

    function pb_value(value) {
        _pb.value = value
    }

    function pb_state(indeterminate) {
        _pb.indeterminate = indeterminate
    }

    function status_message(msg) {
        _status_bar.status_message(msg)
    }

    function popup(title, message, timeout = 2000) {
        _popup.timeout_open(title, message, timeout)
    }

    function get_panel(name) {
        if (name === 'project') {
            return project_panel
        } else if (name === 'registration') {
            return registration_panel
        } else if (name === 'segmentation') {
            return segmentation_panel
        } else if (name === 'panorama') {
            return panorama_panel
        } else if (name === 'descriptive') {
            return descriptive_panel
        }

        return null
    }

    function set_separate_panorama(value) {
        project_panel.set_separate_panorama(value)
    }

    function update_config(config) { // TODO
        let config_json = JSON.parse(config)
        project_panel.update_config(config_json)
        registration_panel.update_config(config_json)
        panorama_panel.update_config(config_json)
    }
}

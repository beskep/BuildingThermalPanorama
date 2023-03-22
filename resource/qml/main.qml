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
    title : '열화상 하자 자동 판정 솔루션'

    FontLoader {
        id : mono
        source : '../font/FiraCode-Regular.ttf'
    }
    FontLoader {
        id : sans
        source : '../font/SourceHanSansKR-Normal.otf'
    }
    FontLoader {
        source : '../font/SourceHanSansKR-Medium.otf'
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

                    ToolTip.visible : hovered
                    ToolTip.delay : 200
                    ToolTip.text : '프로젝트 경로 설정 및 파일 추출'
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
                    text : '파노라마 생성'
                    width : parent.width
                }
                TabButton {
                    text : '파노라마 정합'
                    width : parent.width

                    enabled : separate_panorama
                }
                TabButton {
                    text : '왜곡 보정'
                    width : parent.width
                }
                TabButton {
                    text : '에너지 검진'
                    width : parent.width
                }
                TabButton {
                    text : 'GIS 연동'
                    width : parent.width
                }
            }

            Page {
                Layout.fillHeight : true
                Layout.fillWidth : true

                StackLayout {
                    anchors.fill : parent

                    currentIndex : [
                        0,
                        1,
                        2,
                        3,
                        1, // registration_panel
                        3, // panorama_panel
                        4,
                        5
                    ][tab_bar.currentIndex]

                    onCurrentIndexChanged : {
                        [
                            project_panel,
                            registration_panel,
                            segmentation_panel,
                            panorama_panel,
                            analysis_panel,
                            output_panel
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

                        correction_plot : tab_bar.currentIndex === 5
                    }
                    AnalysisPanel {
                        id : analysis_panel
                    }
                    OutputPanel {
                        id : output_panel
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

    Image {
        source : '../EGSolutionsLogoKR.svg'
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
        } else if (name === 'analysis') {
            return analysis_panel
        }

        return null
    }

    function set_separate_panorama(value) {
        project_panel.set_separate_panorama(value)
    }

    function update_config(config) {
        let config_json = JSON.parse(config)

        project_panel.update_config(config_json)
        registration_panel.update_config(config_json)
        panorama_panel.update_config(config_json)
        output_panel.update_config(config_json)
    }
}

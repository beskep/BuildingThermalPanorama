import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import "Custom"
import "Panorama"

ApplicationWindow {
    id: app

    property ApplicationWindow app: app
    property bool separate_panorama: false
    property string fd: '../../resource/font/'
    property alias project_panel: project_panel
    property alias registration_panel: registration_panel
    property alias segmentation_panel: segmentation_panel
    property alias panorama_panel: panorama_panel
    property alias analysis_panel: analysis_panel
    property alias output_panel: output_panel
    property alias wwr_panel: wwr_panel

    function pb_value(value) {
        _pb.value = value;
    }

    function pb_state(indeterminate) {
        _pb.indeterminate = indeterminate;
    }

    function status_message(msg) {
        _status_bar.text = msg;
    }

    function popup(title, message, timeout = 2000) {
        _popup.timeout_open(title, message, timeout);
    }

    function set_separate_panorama(value) {
        project_panel.set_separate_panorama(value);
    }

    function update_config(config) {
        let config_json = JSON.parse(config);
        project_panel.update_config(config_json);
        registration_panel.update_config(config_json);
        panorama_panel.update_config(config_json);
        output_panel.update_config(config_json);
    }

    width: 1600
    height: 900
    visible: true
    title: '건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램'

    FontLoader {
        id: mono

        source: `${fd}iosevka-ss11-regular.ttf`
    }

    FontLoader {
        id: sans

        source: `${fd}SourceHanSansKR-Normal.otf`
    }

    FontLoader {
        source: `${fd}SourceHanSansKR-Medium.otf`
    }

    FontLoader {
        id: icon

        source: `${fd}MaterialIcons-Regular.ttf`
    }

    ColumnLayout {
        anchors.fill: parent

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true

            VertTabBar {
                id: tab_bar

                Layout.preferredWidth: 200
                Layout.fillHeight: true

                TabButton {
                    text: '프로젝트 설정'
                    width: parent.width
                    ToolTip.visible: hovered
                    ToolTip.delay: 200
                    ToolTip.text: '프로젝트 경로 설정 및 파일 추출'
                }

                TabButton {
                    text: '열·실화상 정합'
                    width: parent.width
                    enabled: !separate_panorama
                }

                TabButton {
                    text: '외피 부위 인식'
                    width: parent.width
                }

                TabButton {
                    text: '파노라마 생성'
                    width: parent.width
                }

                TabButton {
                    text: '파노라마 정합'
                    width: parent.width
                    enabled: separate_panorama
                }

                TabButton {
                    text: '왜곡 보정'
                    width: parent.width
                }

                TabButton {
                    text: '에너지 검진'
                    width: parent.width
                }

                TabButton {
                    text: 'GIS 연동'
                    width: parent.width
                }

                TabButton {
                    text: '창면적비 계산'
                    width: parent.width
                }

                background: Rectangle {
                }

            }

            Page {
                Layout.fillHeight: true
                Layout.fillWidth: true

                StackLayout {
                    anchors.fill: parent
                    // project, registration, segmentation, panorama, registration, panorama, analysis, output, WWR
                    currentIndex: [0, 1, 2, 3, 1, 3, 4, 5, 6][tab_bar.currentIndex]
                    onCurrentIndexChanged: itemAt(currentIndex).init()

                    ProjectPanel {
                        id: project_panel
                    }

                    RegistrationPanel {
                        id: registration_panel

                        // 파노라마 정합 선택 시
                        separate_panorama: separate_panorama
                    }

                    SegmentationPanel {
                        id: segmentation_panel
                    }

                    PanoramaPanel {
                        id: panorama_panel

                        correction_plot: tab_bar.currentIndex === 5
                    }

                    AnalysisPanel {
                        id: analysis_panel
                    }

                    OutputPanel {
                        id: output_panel
                    }

                    WWRPanel {
                        id: wwr_panel
                    }

                }

            }

        }

        ProgressBar {
            id: _pb

            Layout.fillWidth: true
            indeterminate: false
            value: 1
        }

    }

    TimeoutPopup {
        id: _popup
    }

    footer: StatusBar {
        id: _status_bar

        text: '건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램'
    }

}

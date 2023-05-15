import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import Qt.labs.platform 1.1
import QtWebEngine 1.8
import "../Custom"
import Backend 1.0

Pane {
    property int mode: 0 // [analysis, registration, anomaly, report]
    property alias parameter: parameter_table

    function reset() {
        con.display('', 0); // reset plot
        clear_stat();
    }

    function update_image_view(paths) {
        _image_model.clear();
        paths.forEach((path) => _image_model.append({
            "path": path
        }));
    }

    function clear_stat() {
        _stat.table_model.clear();
    }

    function append_stat_row(row) {
        _stat.table_model.appendRow(row);
    }

    function web_view(url) {
        _web_view.url = url;
    }

    width: 1280
    height: 720
    padding: 10

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                spacing: 0

                RowLayout {
                    visible: mode === 0

                    ToolButton {
                        text: '경로 선택'
                        icon: '\ue8a7'
                        onReleased: _folder_dialog.open()
                    }

                    ToolButton {
                        text: '영상 변환'
                        icon: '\ue30d'
                        onReleased: con.qml_command('extract', '열·실화상 추출')
                    }

                }

                RowLayout {
                    visible: mode === 1

                    ToolButton {
                        text: '자동 정합'
                        icon: '\ue663'
                        onReleased: con.qml_command('register', '열·실화상 자동 정합')
                        visible: false
                    }

                    ToolButton {
                        id: _point

                        text: '지점 선택'
                        icon: '\ue55c'
                        down: true
                        onReleased: {
                            down = true;
                            _zoom.down = false;
                        }
                    }

                    ToolButton {
                        id: _zoom

                        text: '확대'
                        icon: '\ue56b'
                        onDownChanged: con.plot_navigation(false, down)
                        onReleased: {
                            down = true;
                            _point.down = false;
                        }
                    }

                    ToolButton {
                        text: '초기 시점'
                        icon: '\ue88a'
                        onReleased: con.plot_navigation(true, false)
                    }

                }

                RowLayout {
                    visible: mode === 2

                    ToolButton {
                        text: '이상 영역 검출'
                        icon: '\ue7ee'
                        onReleased: con.qml_command('segment, detect', '외피 분할 및 열적 이상 영역 검출')
                    }

                }

                RowLayout {
                    visible: mode === 3

                    ToolButton {
                        text: '저장'
                        icon: '\ue161'
                        onReleased: _file_dialog.open()
                    }

                }

            }

        }

        RowLayout {
            Layout.fillHeight: true
            Layout.fillWidth: true
            spacing: 10

            // 영상 목록
            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.preferredWidth: 300
                padding: 5

                ListView {
                    id: _image_view

                    anchors.fill: parent
                    clip: true

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    model: ListModel {
                        id: _image_model
                    }

                    delegate: Pane {
                        Material.elevation: 0
                        width: _image_view.width - 20
                        height: width * 3 / 4 + 10

                        Image {
                            id: _image

                            source: path
                            width: parent.width
                            fillMode: Image.PreserveAspectFit
                        }

                        BrightnessContrast {
                            id: _bc

                            anchors.fill: _image
                            source: _image
                            brightness: 0
                        }

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            onReleased: con.display(path, mode)
                            onEntered: _bc.brightness = -0.25
                            onExited: _bc.brightness = 0
                        }

                    }

                }

            }

            ColumnLayout {
                // project, parameter table
                Pane {
                    Material.elevation: 2
                    Layout.preferredHeight: 300 // TODO 비율
                    Layout.fillWidth: true
                    visible: mode === 0

                    RowLayout {
                        anchors.fill: parent

                        Pane {
                            Material.elevation: 1
                            Layout.fillHeight: true
                            Layout.fillWidth: true

                            ProjectTable {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                        }

                        Pane {
                            Material.elevation: 1
                            Layout.fillHeight: true
                            Layout.fillWidth: true

                            ParameterTable {
                                id: parameter_table

                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                        }

                    }

                }

                // plot
                Pane {
                    Material.elevation: 2
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                    padding: 0
                    visible: mode !== 3

                    FigureCanvas {
                        id: _plot

                        anchors.fill: parent
                        objectName: 'plot'
                        dpi_ratio: Screen.devicePixelRatio
                    }

                }

                // stat table
                Pane {
                    Material.elevation: 2
                    Layout.preferredHeight: 200
                    Layout.fillWidth: true
                    visible: mode === 2

                    StatTable {
                        id: _stat

                        anchors.fill: parent
                    }

                }

                // report
                Pane {
                    Material.elevation: 2
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                    visible: mode === 3

                    WebEngineView {
                        id: _web_view

                        anchors.fill: parent
                    }

                }

            }

        }

    }

    FolderDialog {
        id: _folder_dialog

        onAccepted: con.select_working_dir(folder)
    }

    FileDialog {
        id: _file_dialog

        fileMode: FileDialog.SaveFile
        nameFilters: ['PDF (*.pdf)']
        onAccepted: con.save_report(_web_view.url, file)
    }

}

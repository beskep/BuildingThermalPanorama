import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.qmlmodels 1.0
import "../Custom"
import "../Button" as Btn
import Backend 1.0

Pane {
    property int table_width: 250
    property int tooltip_width: 500

    function init() {
        update_plot(true);
    }

    function update_plot(force) {
        con.wwr_update(_vis.checked ? 'vis' : 'seg', _threshold.value / 100, force);
    }

    function show_wwr(value) {
        table_model.clear();
        table_model.appendRow(value);
    }

    function save_wwr() {
        con.wwr_save(JSON.stringify(table_model.rows[0]));
    }

    width: 1280
    height: 720
    padding: 10

    Popup {
        id: _option

        anchors.centerIn: Overlay.overlay
        Material.elevation: 5
        padding: 0
        height: _option_content.implicitHeight

        ColumnLayout {
            id: _option_content

            anchors.fill: parent

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: 20
                Layout.minimumWidth: 450
                Layout.maximumWidth: 750
                spacing: 20

                Label {
                    Layout.fillWidth: true
                    font.pointSize: 16
                    font.weight: Font.Medium
                    text: '창면적비 설정'
                }

                RowLayout {
                    Label {
                        font.bold: true
                        text: '외피 가림 임계치'
                    }

                    FloatSpinBox {
                        id: _threshold

                        value: 10
                        from: 0
                        to: 100
                        stepSize: 5
                        onValueChanged: update_plot(false)

                        ToolTip {
                            text: '외피 외 면적의 비중이 설정치 이상인 층은 창면적비 추정 데이터에서 제외합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                }

            }

        }

    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                Btn.ToolButton {
                    text: '저장'
                    icon: '\ue161'
                    onReleased: save_wwr()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: '창면적비 저장'
                }

                ToolSeparator {
                }

                Btn.Navigation {
                    index: 8
                }

                ToolSeparator {
                }

                Btn.Setting {
                    onReleased: _option.open()
                }

                Btn.Help {
                    // TODO

                }

            }

        }

        Pane {
            Material.elevation: 2
            Layout.fillHeight: true
            Layout.fillWidth: true
            padding: 0

            FigureCanvas {
                id: plot

                anchors.fill: parent
                objectName: 'wwr_plot'
                dpi_ratio: Screen.devicePixelRatio
            }

            // 툴바
            Pane {
                anchors.left: parent.left
                anchors.top: parent.top
                Material.elevation: 1
                padding: 0
                leftPadding: 5

                RowLayout {
                    RowLayout {
                        visible: _expand.expanded

                        RadioButton {
                            id: _vis

                            checked: true
                            text: '실화상'
                            onCheckedChanged: update_plot(false)
                        }

                        RadioButton {
                            text: '부위 인식'
                        }

                    }

                    Btn.Expand {
                        id: _expand

                        padding: 0
                    }

                }

            }

        }

        Pane {
            Material.elevation: 2
            Layout.preferredHeight: 100
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent

                ColumnLayout {
                    spacing: 0
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    HorizontalHeaderView {
                        syncView: table_view
                        Layout.fillWidth: true

                        model: ListModel {
                            ListElement {
                                name: '벽 픽셀 수 (ⓐ)'
                            }

                            ListElement {
                                name: '창 픽셀 수 (ⓑ)'
                            }

                            ListElement {
                                name: '외피 픽셀 수 (ⓒ=ⓐ+ⓑ)'
                            }

                            ListElement {
                                name: '창면적비 (ⓑ/ⓒ)'
                            }

                        }

                        delegate: Rectangle {
                            implicitHeight: 40
                            implicitWidth: table_width
                            color: '#eeeeee'

                            Label {
                                text: name
                                horizontalAlignment: Text.AlignHCenter
                                anchors.centerIn: parent
                            }

                        }

                    }

                    TableView {
                        id: table_view

                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        columnSpacing: 1
                        rowSpacing: 1
                        boundsBehavior: Flickable.StopAtBounds

                        model: TableModel {
                            id: table_model

                            rows: [{
                                "wall": '-',
                                "window": '-',
                                "envelope": '-',
                                "wwr": '-'
                            }]

                            TableModelColumn {
                                display: 'wall'
                            }

                            TableModelColumn {
                                display: 'window'
                            }

                            TableModelColumn {
                                display: 'envelope'
                            }

                            TableModelColumn {
                                display: 'wwr'
                            }

                        }

                        delegate: Rectangle {
                            implicitHeight: 40
                            implicitWidth: table_width

                            Label {
                                text: display
                                anchors.centerIn: parent
                            }

                        }

                    }

                }

            }

        }

    }

}

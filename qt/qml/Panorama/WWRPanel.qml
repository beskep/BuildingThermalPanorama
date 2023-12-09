import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import Qt.labs.qmlmodels 1.0
import "../Custom"
import Backend 1.0

Pane {
    property int table_width: 250

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

    width: 1280
    height: 720
    padding: 10

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                ToolRadioButton {
                    id: _vis

                    checked: true
                    text: '실화상'
                    onCheckedChanged: update_plot(false)
                }

                ToolRadioButton {
                    text: '부위 인식'
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

        }

        Pane {
            Material.elevation: 2
            Layout.preferredHeight: 160
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent

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
                    }

                }

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

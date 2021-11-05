import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import Qt.labs.qmlmodels 1.0

import 'Custom'
import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'descriptive_panel'

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            spacing : 0

            ToolButton {
                text : qsTr('온도 분포 분석')
                icon : '\ue01d'

                onReleased : con.dist_plot()

                ToolTip.visible : hovered
                ToolTip.delay : 500
                ToolTip.text : qsTr('외피 열화상의 부위별 온도 분포 분석')
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillHeight : true
            Layout.fillWidth : true
            padding : 0

            FigureCanvas {
                id : plot
                anchors.fill : parent
                objectName : 'dist_plot'
                Layout.fillHeight : true
                Layout.fillWidth : true
                dpi_ratio : Screen.devicePixelRatio
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillWidth : true
            Layout.preferredHeight : 150

            ColumnLayout {
                anchors.fill : parent
                spacing : 0

                HorizontalHeaderView {
                    syncView : table_view
                    Layout.fillWidth : true

                    model : ListModel {
                        ListElement {
                            name : '클래스'
                        }
                        ListElement {
                            name : '평균'
                        }
                        ListElement {
                            name : '표준편차'
                        }
                        ListElement {
                            name : 'Q1'
                        }
                        ListElement {
                            name : '중위수'
                        }
                        ListElement {
                            name : 'Q3'
                        }
                    }
                    delegate : Rectangle {
                        implicitHeight : 40
                        implicitWidth : 120
                        color : '#eeeeee'

                        Label {
                            text : name
                            anchors.centerIn : parent
                        }
                    }
                }

                TableView {
                    id : table_view
                    columnSpacing : 1
                    rowSpacing : 1
                    boundsBehavior : Flickable.StopAtBounds

                    Layout.fillWidth : true
                    Layout.fillHeight : true

                    model : TableModel {
                        id : table_model

                        TableModelColumn {
                            display : 'class'
                        }
                        TableModelColumn {
                            display : 'avg'
                        }
                        TableModelColumn {
                            display : 'std'
                        }
                        TableModelColumn {
                            display : 'q1'
                        }
                        TableModelColumn {
                            display : 'median'
                        }
                        TableModelColumn {
                            display : 'q3'
                        }

                        rows : []
                    }

                    delegate : Rectangle {
                        implicitHeight : 40
                        implicitWidth : 120

                        Label {
                            text : display
                            anchors.centerIn : parent
                        }
                    }
                }
            }
        }
    }

    function init() {}

    function clear_table() {
        table_model.clear()
    }

    function add_table_row(row) {
        table_model.appendRow(row)
    }
}

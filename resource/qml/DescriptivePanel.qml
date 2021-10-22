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

                HorizontalHeaderView {
                    syncView : table_view
                    Layout.fillWidth : true

                    model : ListModel {
                        ListElement {
                            name : 'col1'
                        }
                        ListElement {
                            name : 'col2'
                        }
                    }
                    delegate : Label {
                        text : name
                    }
                }
                TableView {
                    id : table_view
                    columnSpacing : 1
                    rowSpacing : 1

                    Layout.fillWidth : true
                    Layout.fillHeight : true

                    model : TableModel {
                        id : table_model
                    }

                    delegate : Rectangle {
                        implicitHeight : 50
                        implicitWidth : 100

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
}

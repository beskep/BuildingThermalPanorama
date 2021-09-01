import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12
import QtQuick.Window 2.12

import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'registration_panel'

    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Button {
                text : qsTr('전체 자동 정합')
            }
            Button {
                text : qsTr('자동 정합')
            }
            Button {
                text : qsTr('수동 정합')
            }
            Rectangle {
                width : 10
            }
            Button {
                text : qsTr('저장')
            }
            Button {
                text : qsTr('취소')
            }
        }

        RowLayout {
            spacing : 10

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.preferredWidth : 200
            }
            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true

                ColumnLayout {
                    anchors.fill : parent

                    RegistrationCanvas {
                        id : plot
                        objectName : 'registration_plot'
                        Layout.fillHeight : true
                        Layout.fillWidth : true
                        dpi_ratio : Screen.devicePixelRatio
                    }
                }
            }
        }
    }
}

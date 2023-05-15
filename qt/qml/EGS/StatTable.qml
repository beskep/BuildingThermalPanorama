import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import Qt.labs.qmlmodels 1.0

Item {
    property alias table_model: table_model
    property var titles: ['클래스', '평균', '표준편차', 'Q1', '중위수', 'Q3']

    TableModel {
        id: table_model

        TableModelColumn {
            display: 'class'
        }

        TableModelColumn {
            display: 'avg'
        }

        TableModelColumn {
            display: 'std'
        }

        TableModelColumn {
            display: 'q1'
        }

        TableModelColumn {
            display: 'median'
        }

        TableModelColumn {
            display: 'q3'
        }

    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        HorizontalHeaderView {
            syncView: table_view
            model: titles.length

            delegate: Rectangle {
                implicitHeight: 50
                implicitWidth: 150
                color: '#eeeeee'

                Label {
                    anchors.centerIn: parent
                    horizontalAlignment: Text.AlignHCenter
                    text: titles[modelData]
                }

            }

        }

        TableView {
            id: table_view

            columnSpacing: 1
            rowSpacing: 1
            boundsBehavior: Flickable.StopAtBounds
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: table_model

            delegate: Rectangle {
                implicitHeight: 40
                implicitWidth: 150

                Label {
                    anchors.centerIn: parent
                    text: display
                }

            }

        }

    }

}

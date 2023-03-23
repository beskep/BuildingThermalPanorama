import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Pane {
    property alias text : status_text.text

    height : 32
    horizontalPadding : 20
    verticalPadding : 0

    background : Rectangle {
        color : '#E0E0E0'
    }

    RowLayout {
        anchors.fill : parent

        Label {
            id : status_text

            Layout.alignment : Qt.AlignVCenter
            font.pointSize : 11
            color : "#212121"
        }

        Label {
            Layout.fillWidth : true
        }
    }
}

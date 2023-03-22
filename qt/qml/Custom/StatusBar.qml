import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Pane {
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

            text : '열화상 하자 자동 판정 솔루션'
        }

        Label {
            Layout.fillWidth : true
        }
    }

    function status_message(msg) {
        status_text.text = msg
    }
}

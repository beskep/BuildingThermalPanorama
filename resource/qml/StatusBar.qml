import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12


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
            font.pointSize : 10
            color : "#212121"

            text : '건물 에너지 검진을 위한 열화상 파노라마 영상처리 프로그램'
        }

        Label {
            Layout.fillWidth : true
        }
    }
}

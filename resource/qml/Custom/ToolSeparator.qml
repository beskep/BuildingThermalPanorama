import QtQuick 2.9
import QtQuick.Templates 2.2 as T
import QtQuick.Controls.Material 2.2


T.ToolSeparator {
    id: control

    implicitWidth: Math.max(background ? background.implicitWidth : 0, contentItem.implicitWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(background ? background.implicitHeight : 0, contentItem.implicitHeight + topPadding + bottomPadding)

    leftPadding: vertical ? 12 : 5
    rightPadding: vertical ? 12 : 5
    topPadding: vertical ? 5 : 12
    bottomPadding: vertical ? 5 : 12

    contentItem: Rectangle {
        implicitWidth: vertical ? 1 : 30
        implicitHeight: vertical ? 30 : 1
        color: '#e0e0e0'
    }
}

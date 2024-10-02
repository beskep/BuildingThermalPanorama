import QtQuick 2.12
import QtQuick.Templates 2.12 as T
import QtQuick.Controls.Material 2.12
import QtQuick.Controls.Material.impl 2.12

T.ProgressBar {
    id: control

    property real bar_height: 4
    property real bar_alpha: 1
    property real background_alpha: 0.25

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset, implicitContentWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(implicitBackgroundHeight + topInset + bottomInset, implicitContentHeight + topPadding + bottomPadding)

    contentItem: ProgressBarImpl {
        implicitHeight: bar_height
        scale: control.mirrored ? -1 : 1
        color: Qt.rgba(control.Material.accentColor.r, control.Material.accentColor.g, control.Material.accentColor.b, bar_alpha)
        progress: control.position
        indeterminate: control.visible && control.indeterminate
    }

    background: Rectangle {
        implicitWidth: 200
        implicitHeight: bar_height
        y: (control.height - height) / 2
        height: bar_height
        color: Qt.rgba(control.Material.accentColor.r, control.Material.accentColor.g, control.Material.accentColor.b, background_alpha)
    }

}

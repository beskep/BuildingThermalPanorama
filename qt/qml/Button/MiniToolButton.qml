import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import "../Button"

ToolButton {
    leftPadding: 5
    rightPadding: 5
    text_color: '#212121'
    ripple_color: '#B2DFDB'
    background.implicitHeight: 32
    ToolTip.visible: hovered
    ToolTip.delay: 200
}

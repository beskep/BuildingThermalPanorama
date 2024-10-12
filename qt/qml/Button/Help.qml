import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import "../Button"

ToolButton {
    text: '도움말'
    text_color: '#A0FFFFFF'
    icon: '\ue88e'
    ToolTip.visible: hovered
    ToolTip.delay: 200
}

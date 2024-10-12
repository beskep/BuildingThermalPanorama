import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import "../Button"

ToolButton {
    text: '설정'
    text_color: '#A0FFFFFF'
    icon: '\ue8b8'
    ToolTip.visible: hovered
    ToolTip.delay: 200
}

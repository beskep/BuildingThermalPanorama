import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import "../Button"

ToolButton {
    text: '폴더 열기'
    icon: '\ue2c8'
    ToolTip.visible: hovered
    ToolTip.delay: 200
    ToolTip.text: '작업 결과 폴더 열기'
}

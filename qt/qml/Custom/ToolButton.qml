import QtQuick 2.9
import QtQuick.Templates 2.2 as T
import QtQuick.Controls.Material 2.2
import QtQuick.Controls.Material.impl 2.2
import QtQuick.Layouts 1.15

T.ToolButton {
    id: control

    property alias icon: _icon.text
    property alias icon2: _icon2.text
    property alias text_size: _text.font.pointSize
    property var text_color: '#FFF'

    implicitWidth: Math.max(background ? background.implicitWidth : 0, contentItem.implicitWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(background ? background.implicitHeight : 0, contentItem.implicitHeight + topPadding + bottomPadding)
    baselineOffset: contentItem.y + contentItem.baselineOffset
    padding: 4
    leftPadding: 15
    rightPadding: 15

    contentItem: RowLayout {
        Layout.alignment: Qt.AlignVCenter

        Text {
            id: _icon

            text: ''
            font.family: 'Material Symbols Outlined'
            font.pointSize: _text.font.pointSize + 2
            color: (!control.enabled ? control.Material.hintTextColor : (control.checked || control.highlighted) ? control.Material.accent : text_color)
            elide: Text.ElideRight
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }

        Text {
            id: _text

            text: control.text
            font: control.font
            color: (!control.enabled ? control.Material.hintTextColor : (control.checked || control.highlighted) ? control.Material.accent : text_color)
            elide: Text.ElideRight
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }

        Text {
            id: _icon2

            text: ''
            font.family: 'Material Symbols Outlined'
            font.pointSize: _text.font.pointSize + 2
            color: (!control.enabled ? control.Material.hintTextColor : (control.checked || control.highlighted) ? control.Material.accent : text_color)
            elide: Text.ElideRight
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }

    }

    background: Ripple {
        readonly property bool square: control.contentItem.width <= control.contentItem.height

        implicitWidth: 48
        implicitHeight: 48
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
        clip: !square
        width: square ? parent.height / 2 : parent.width
        height: square ? parent.height / 2 : parent.height
        pressed: control.pressed
        anchor: control
        active: control.enabled && (control.down || control.visualFocus || control.hovered)
        color: control.Material.rippleColor
    }

}

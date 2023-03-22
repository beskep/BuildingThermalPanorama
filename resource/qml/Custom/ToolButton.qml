import QtQuick 2.9
import QtQuick.Templates 2.2 as T
import QtQuick.Controls.Material 2.2
import QtQuick.Controls.Material.impl 2.2
import QtQuick.Layouts 1.15


T.ToolButton {
    id : control

    implicitWidth : Math.max( //
        background ? background.implicitWidth : 0, //
        contentItem.implicitWidth + leftPadding + rightPadding)
    implicitHeight : Math.max( //
        background ? background.implicitHeight : 0, //
        contentItem.implicitHeight + topPadding + bottomPadding)
    baselineOffset : contentItem.y + contentItem.baselineOffset

    padding : 4
    leftPadding : 15
    rightPadding : 15

    property alias icon : _icon.text

    contentItem : RowLayout {
        Layout.alignment : Qt.AlignVCenter

        Text {
            id : _icon
            text : ''
            font.family : 'Material Icons'
            font.pointSize : _text.font.pixelSize
            color : (!control.enabled ? control.Material.hintTextColor : //
                    (control.checked || control.highlighted) ? control.Material.accent : //
                    '#fff')
            elide : Text.ElideRight
            horizontalAlignment : Text.AlignHCenter
            verticalAlignment : Text.AlignVCenter
        }
        Text {
            id : _text
            text : control.text
            font : control.font
            color : (!control.enabled ? control.Material.hintTextColor : //
                    (control.checked || control.highlighted) ? control.Material.accent : //
                    '#fff')
            elide : Text.ElideRight
            horizontalAlignment : Text.AlignHCenter
            verticalAlignment : Text.AlignVCenter
        }
    }

    background : Ripple {
        implicitWidth : 48
        implicitHeight : 48

        readonly property bool square : control.contentItem.width <= control.contentItem.height

        x : (parent.width - width) / 2
        y : (parent.height - height) / 2
        clip : !square
        width : square ? parent.height / 2 : parent.width
        height : square ? parent.height / 2 : parent.height
        pressed : control.pressed
        anchor : control
        active : control.enabled && (control.down || control.visualFocus || control.hovered)
        color : control.Material.rippleColor
    }
}

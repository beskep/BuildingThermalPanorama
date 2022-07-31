import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


RoundButton {
    property var expanded: true;

    flat : true
    font.family : 'Material Icons'
    font.pixelSize : 24

    text : expanded ? '\ue5dc' : '\ue5dd'

    onReleased : {
        expanded = ! expanded
    }
}

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15


Slider {
    id : control

    from : -100
    value : 0
    to : 100
    stepSize : 1

    snapMode : Slider.SnapAlways

    background : Rectangle {
        x : control.leftPadding
        y : control.topPadding + control.availableHeight / 2 - height / 2
        implicitWidth : 200
        implicitHeight : 4
        width : control.availableWidth
        height : implicitHeight
        color : Material.color(control.Material.accentColor, Material.Shade100)

        // Rectangle {
        //     width : control.visualPosition * parent.width
        //     height : parent.height
        //     color : control.Material.accentColor
        // }
    }
}

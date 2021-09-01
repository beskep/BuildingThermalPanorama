import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.12
import QtQuick.Window 2.12

import QtQuick.Templates 2.12 as T


T.TabBar {
    id : control

    // implicitWidth : Math.max(background ? background.implicitWidth : 0,
    //                          contentWidth + leftPadding + rightPadding)
    // implicitHeight : Math.max(background ? background.implicitHeight : 0,
    //                           contentHeight + topPadding + bottomPadding)

    spacing : 1
    contentHeight : 40

    contentItem : ListView {
        model : control.contentModel
        currentIndex : control.currentIndex

        spacing : control.spacing
        orientation : ListView.Vertical
        boundsBehavior : Flickable.StopAtBounds
        flickableDirection : Flickable.AutoFlickIfNeeded
        snapMode : ListView.SnapToItem

        highlightMoveDuration : 0
        highlightRangeMode : ListView.ApplyRange
        preferredHighlightBegin : 40
        preferredHighlightEnd : width - 40
    }

    background : Rectangle {}
}

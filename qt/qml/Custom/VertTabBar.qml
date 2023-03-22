import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import QtQuick.Templates 2.12 as T


T.TabBar {
    id : control

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

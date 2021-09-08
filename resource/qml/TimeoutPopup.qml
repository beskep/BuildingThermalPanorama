import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

Popup {
    id : _popup

    anchors.centerIn : Overlay.overlay

    property var steps: 200.0

    padding : 0
    implicitWidth : 400
    implicitHeight : 250

    ColumnLayout {
        anchors.fill : parent

        ProgressBar {
            id : _pb
            Layout.fillWidth : true

            value : 0.0

            onValueChanged : {
                if (value >= 1.0) {
                    _timer.stop()
                    _popup.close()
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth : true
            Layout.fillHeight : true
            Layout.margins : 20
            spacing : 10

            Label {
                id : _title
                Layout.fillWidth : true

                font.pointSize : 16
                font.weight : Font.Medium

                text : 'Information'
            }
            Label {
                id : _message
                Layout.fillWidth : true
                Layout.fillHeight : true

                font.pointSize : 14
                wrapMode : Label.WordWrap

                text : '[Default message]'
            }
            Button {
                Layout.alignment : Qt.AlignRight | Qt.AlignBottom
                flat : true

                text : 'OK'

                onClicked : {
                    _popup.close()
                }
            }
        }
    }

    Timer {
        id : _timer
        interval : 20
        running : false
        repeat : true
        onTriggered : {
            _pb.value += 1 / steps
        }
    }

    function timeout_open(title, message, timeout = 2000) {
        _title.text = title
        _message.text = message
        _timer.interval = timeout / steps
        _pb.value = 0.0

        open()
        _timer.start()
    }
}

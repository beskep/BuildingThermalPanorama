import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Popup {
    id : _popup

    property var steps: 200.0;

    anchors.centerIn : Overlay.overlay
    Material.elevation : 5
    padding : 0
    height : _content.implicitHeight

    TextMetrics {
        id : _metrics
        font.pointSize : 12
        text : ''
    }

    ColumnLayout {
        id : _content
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
            Layout.minimumWidth : 200
            Layout.maximumWidth : 750
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

                font.pointSize : 12
                wrapMode : Label.Wrap

                text : _metrics.text
                onTextChanged : {
                    if (_metrics.width <= 200) {
                        Layout.preferredWidth = 200
                    } else {
                        Layout.preferredWidth = Math.ceil(Math.sqrt(_metrics.width * _message.height * 2))
                    }
                }
            }
            Button {
                Layout.alignment : Qt.AlignRight | Qt.AlignBottom
                flat : true

                text : 'OK'

                onClicked : _popup.close()
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
        _metrics.text = message
        _timer.interval = timeout / steps
        _pb.value = 0.0

        open()
        _timer.start()
    }
}

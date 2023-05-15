import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

ColumnLayout {
    function display(texts) {
        for (var i = 0; i < repeater.model.length; i++) {
            let item = repeater.itemAt(i);
            item.label = texts[i];
        }
    }

    spacing: 0

    Repeater {
        id: repeater

        model: ['방사율', '반사 겉보기 온도', '측정 거리', '상대 습도', '대기 온도']

        RowLayout {
            property string label: ''

            Layout.preferredHeight: 42
            spacing: 0

            Label {
                font.bold: true
                Layout.preferredWidth: 150
                Layout.fillHeight: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: modelData

                background: Rectangle {
                    border.width: 1
                    border.color: '#E0E0E0'
                }

            }

            Label {
                Layout.preferredWidth: 200
                Layout.fillHeight: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: parent.label

                background: Rectangle {
                    border.width: 1
                    border.color: '#E0E0E0'
                }

            }

        }

    }

}

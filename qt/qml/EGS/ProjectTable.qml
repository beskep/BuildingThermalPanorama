import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

ColumnLayout {
    property var key: ['location', 'building', 'date', 'part']
    property var display: ['측정 위치', '건물 명칭', '측정 날짜', '측정 부위']

    spacing: 0

    Repeater {
        model: key.length

        Pane {
            Layout.preferredHeight: 60

            RowLayout {
                spacing: 16

                Label {
                    id: label

                    Layout.preferredWidth: 100
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font.bold: true
                    text: display[modelData]
                }

                TextField {
                    id: text_field

                    Layout.preferredWidth: 240
                    placeholderText: '프로젝트 정보 입력'
                    onEditingFinished: {
                        let index = display.indexOf(label.text);
                        con.set_project_data(key[index], text_field.text);
                    }
                }

            }

            background: Rectangle {
                border.width: 1
                border.color: '#E0E0E0'
            }

        }

    }

}

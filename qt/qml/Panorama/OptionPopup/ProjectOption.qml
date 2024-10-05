import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../../Custom"

Popup {
    id: _popup

    property string _warning: ' (<u>설정 변경 시 부위 인식 및 파노라마 생성 결과 초기화</u>)'
    property var _config: {
        "panorama": null
    }

    function reset() {
        let conf = _config['panorama'];
        if (!conf)
            return ;

        _separate.checked = conf['separate'];
    }

    function configure() {
        // TODO test
        _config = {
            "panorama": {
                "separate": _separate.checked
            }
        };
        con.configure(JSON.stringify(_config));
        app.separate_panorama = _separate.checked;
    }

    function update_config(config) {
        // TODO test
        _config['panorama'] = config['panorama'];
        reset();
    }

    anchors.centerIn: Overlay.overlay
    Material.elevation: 5
    padding: 0
    height: _content.implicitHeight

    ColumnLayout {
        id: _content

        anchors.fill: parent

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            Layout.minimumWidth: 300
            Layout.maximumWidth: 750
            spacing: 20

            Label {
                id: _title

                Layout.fillWidth: true
                font.pointSize: 16
                font.weight: Font.Medium
                text: '프로젝트 설정'
            }

            ColumnLayout {
                spacing: 0

                Label {
                    Layout.fillWidth: true
                    font.weight: Font.Medium
                    font.pointSize: 13
                    text: '열·실화상 파노라마 생성 설정'
                }

                RowLayout {
                    Label {
                        Layout.fillWidth: true
                        text: '별도 생성'
                    }

                    RadioButton {
                        id: _separate

                        checked: true
                        ToolTip.visible: hovered
                        ToolTip.text: '열·실화상 파노라마 별도 생성 후 두 파노라마를 정합.' + _warning
                    }

                    Rectangle {
                        width: 50
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '동시 생성'
                    }

                    RadioButton {
                        id: _simultaneously

                        checked: false
                        ToolTip.visible: hovered
                        ToolTip.text: '열·실화상 정합 후 파노라마 동시 생성.' + _warning
                    }

                }

            }

            RowLayout {
                Layout.alignment: Qt.AlignRight | Qt.AlignBottom

                Button {
                    flat: true
                    text: 'Cancel'
                    onClicked: {
                        reset();
                        _popup.close();
                    }
                }

                Button {
                    flat: true
                    text: 'OK'
                    onClicked: {
                        configure();
                        _popup.close();
                    }
                }

            }

        }

    }

}

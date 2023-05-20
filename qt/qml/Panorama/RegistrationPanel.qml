import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import "../Custom"
import "OptionPopup"
import Backend 1.0

Pane {
    property bool separate_panorama: false

    function init() {
        con.rgst_reset();
        if (separate_panorama)
            con.rgst_pano_draw();

    }

    function update_image_view(paths) {
        image_model.clear();
        paths.forEach((path) => image_model.append({
            "path": path
        }));
    }

    function update_config(config) {
        _option.update_config(config);
        separate_panorama = config['panorama']['separate'];
    }

    width: 1280
    height: 720
    padding: 10
    objectName: 'registration_panel'

    RegistrationOption {
        id: _option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                spacing: 0

                ToolButton {
                    text: qsTr('자동 정합')
                    icon: '\ue663'
                    onReleased: con.command('register')
                    visible: !separate_panorama
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('전체 열화상·실화상 자동 정합')
                }

                ToolSeparator {
                    visible: !separate_panorama
                }

                ToolButton {
                    id: _point

                    text: qsTr('지점 선택')
                    icon: '\ue55c'
                    down: true
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('열화상과 실화상의 대응되는 네 지점을 선택해서 수동으로 정합')
                    onReleased: {
                        down = true;
                        _zoom.down = false;
                    }
                }

                ToolButton {
                    id: _zoom

                    text: qsTr('확대')
                    icon: '\ue56b'
                    onDownChanged: con.rgst_zoom(down)
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('정밀한 지점 선택을 위해 확대할 영역 지정')
                    onReleased: {
                        down = true;
                        _point.down = false;
                    }
                }

                ToolSeparator {
                }

                ToolButton {
                    text: qsTr('초기 시점')
                    icon: '\ue88a'
                    onReleased: con.rgst_home()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('영역 확대를 취소하고 전체 영상 표시')
                }

                ToolButton {
                    text: qsTr('저장')
                    icon: '\ue161'
                    onReleased: con.rgst_save()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('대상 영상의 수동 정합 결과 저장')
                }

                ToolButton {
                    text: qsTr('취소')
                    icon: '\ue14a'
                    onReleased: con.rgst_reset()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('수동 정합 취소')
                }

                RowLayout {
                    visible: !separate_panorama

                    ToolSeparator {
                    }

                    ToolButton {
                        text: qsTr('설정')
                        icon: '\ue8b8'
                        onReleased: _option.open()
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('자동 열·실화상 정합 설정')
                    }

                }

            }

        }

        RowLayout {
            spacing: 10

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.preferredWidth: 300
                padding: 5
                visible: !separate_panorama

                ListView {
                    id: image_view

                    anchors.fill: parent
                    clip: true

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    model: ListModel {
                        id: image_model
                    }

                    delegate: Pane {
                        Material.elevation: 0
                        width: image_view.width - 20
                        height: width * 3 / 4 + 10

                        Image {
                            id: _image

                            source: path
                            width: parent.width
                            fillMode: Image.PreserveAspectFit
                        }

                        BrightnessContrast {
                            id: _bc

                            anchors.fill: _image
                            source: _image
                            brightness: 0
                        }

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            onReleased: con.rgst_plot(path)
                            onEntered: _bc.brightness = -0.25
                            onExited: _bc.brightness = 0
                        }

                    }

                }

            }

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.fillWidth: true
                padding: 0

                FigureCanvas {
                    id: plot

                    anchors.fill: parent
                    objectName: 'registration_plot'
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                    dpi_ratio: Screen.devicePixelRatio
                }

                GridLayout {
                    anchors.fill: parent

                    RowLayout {
                        Layout.alignment: Qt.AlignRight | Qt.AlignBottom
                        Layout.rightMargin: 20

                        Label {
                            text: qsTr('그리드')
                        }

                        CheckBox {
                            padding: 0
                            checkState: Qt.Unchecked
                            onCheckStateChanged: con.rgst_set_grid(checkState === Qt.Checked)
                        }

                    }

                }

            }

        }

    }

}

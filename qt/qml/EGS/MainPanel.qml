import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import Qt.labs.platform 1.1
import Qt.labs.qmlmodels 1.0

import '../Custom'
import Backend 1.0


Pane {
    property var mode: 0; // [analysis, registration, anomaly, report]

    width : 1280
    height : 720
    padding : 10

    ListModel {
        id : table_header_model

        ListElement {
            name : '클래스'
        }
        ListElement {
            name : '평균'
        }
        ListElement {
            name : '표준편차'
        }
        ListElement {
            name : 'Q1'
        }
        ListElement {
            name : '중위수'
        }
        ListElement {
            name : 'Q3'
        }
    }

    TableModel {
        id : table_model

        TableModelColumn {
            display : 'class'
        }
        TableModelColumn {
            display : 'avg'
        }
        TableModelColumn {
            display : 'std'
        }
        TableModelColumn {
            display : 'q1'
        }
        TableModelColumn {
            display : 'median'
        }
        TableModelColumn {
            display : 'q3'
        }
    }

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 0

                RowLayout {
                    visible : mode === 0

                    ToolButton {
                        text : '경로 선택'
                        icon : '\ue8a7'
                        onReleased : _dialog.open()
                    }

                    ToolButton {
                        text : '영상 변환'
                        icon : '\ue30d'
                        onReleased : con.qml_command('extract', '열·실화상 추출')
                    }
                }

                RowLayout {
                    visible : mode === 1
                    ToolButton {
                        text : '자동 정합'
                        icon : '\ue663'
                        onReleased : con.qml_command( //
                                'register', '열·실화상 자동 정합')

                        visible : false
                    }
                    ToolButton {
                        id : _point
                        text : '지점 선택'
                        icon : '\ue55c'

                        down : true
                        onReleased : {
                            down = true;
                            _zoom.down = false;
                        }
                    }
                    ToolButton {
                        id : _zoom
                        text : '확대'
                        icon : '\ue56b'

                        onReleased : {
                            down = true;
                            _point.down = false;
                        }
                        onDownChanged : con.plot_navigation(false, down)
                    }
                    ToolButton {
                        text : '초기 시점'
                        icon : '\ue88a'
                        onReleased : con.plot_navigation(true, false)
                    }
                }

                RowLayout {
                    visible : mode === 2

                    ToolButton {
                        text : '이상 영역 검출'
                        icon : '\ue7ee'
                        onReleased : con.qml_command( //
                                'segment, detect', //
                                '외피 분할 및 열적 이상 영역 검출')
                    }
                }

                RowLayout {
                    visible : mode === 3

                    ToolButton {
                        text : '보고서 생성'
                        icon : '\ue873'
                    }
                    ToolButton {
                        text : '저장'
                        icon : '\ue161'
                    }
                }
            }
        }

        RowLayout {
            Layout.fillHeight : true
            Layout.fillWidth : true
            spacing : 10

            // 영상 목록
            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.preferredWidth : 300
                padding : 5

                ListView {
                    id : _image_view

                    anchors.fill : parent
                    clip : true

                    ScrollBar.vertical : ScrollBar {
                        policy : ScrollBar.AsNeeded
                    }

                    model : ListModel {
                        id : _image_model
                    }

                    delegate : Pane {
                        Material.elevation : 0
                        width : _image_view.width - 20
                        height : width * 3 / 4 + 10

                        Image {
                            id : _image
                            source : path
                            width : parent.width
                            fillMode : Image.PreserveAspectFit
                        }

                        BrightnessContrast {
                            id : _bc
                            anchors.fill : _image
                            source : _image
                            brightness : 0
                        }

                        MouseArea {
                            anchors.fill : parent
                            hoverEnabled : true

                            onReleased : con.plot(path, mode)
                            onEntered : _bc.brightness = -0.25
                            onExited : _bc.brightness = 0
                        }
                    }
                }
            }

            ColumnLayout {
                Pane {
                    Material.elevation : 2
                    Layout.fillHeight : true
                    Layout.fillWidth : true
                    padding : 0
                    visible : mode !== 3

                    // plot
                    FigureCanvas {
                        id : _plot
                        anchors.fill : parent

                        objectName : 'plot'
                        dpi_ratio : Screen.devicePixelRatio
                    }
                }

                Pane {
                    Material.elevation : 2
                    Layout.preferredHeight : 200
                    Layout.fillWidth : true
                    visible : mode === 2

                    // table
                    ColumnLayout {
                        anchors.fill : parent
                        spacing : 0

                        HorizontalHeaderView {
                            syncView : table_view

                            model : table_header_model
                            delegate : Rectangle {
                                implicitHeight : 50
                                implicitWidth : 150
                                color : '#eeeeee'

                                Label {
                                    anchors.centerIn : parent
                                    horizontalAlignment : Text.AlignHCenter
                                    text : name
                                }
                            }
                        }

                        TableView {
                            id : table_view
                            columnSpacing : 1
                            rowSpacing : 1
                            boundsBehavior : Flickable.StopAtBounds

                            Layout.fillWidth : true
                            Layout.fillHeight : true

                            model : table_model

                            delegate : Rectangle {
                                implicitHeight : 40
                                implicitWidth : 150

                                Label {
                                    anchors.centerIn : parent
                                    text : display
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    FolderDialog {
        id : _dialog
        onAccepted : con.select_working_dir(folder)
    }

    function update_image_view(paths) {
        _image_model.clear();
        paths.forEach(path => _image_model.append({'path': path}));
    }

    function clear_table() {
        table_model.clear()
    }

    function append_table_row(row) {
        table_model.appendRow(row)
    }
}

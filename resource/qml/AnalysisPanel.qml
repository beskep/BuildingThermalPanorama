import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import Qt.labs.qmlmodels 1.0

import 'Custom'
import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'analysis_panel'

    property real min_temperature : 0.0
    property real max_temperature : 1.0

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            spacing : 0
        }

        RowLayout {
            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                Layout.fillWidth : true
                padding : 0

                FigureCanvas {
                    id : plot
                    anchors.fill : parent
                    objectName : 'analysis_plot'
                    Layout.fillHeight : true
                    Layout.fillWidth : true
                    dpi_ratio : Screen.devicePixelRatio
                }
            }

            Pane {
                Material.elevation : 2
                Layout.fillHeight : true
                contentWidth : 110

                ColumnLayout {
                    anchors.fill : parent

                    Label {
                        text : '표시 온도 범위'
                    }

                    RowLayout {
                        RangeSlider {
                            id : _slider
                            Layout.fillHeight : true

                            orientation : Qt.Vertical
                            from : min_temperature
                            to : max_temperature
                            stepSize : 0.1
                            // stepSize : 1.0
                            snapMode : RangeSlider.SnapAlways

                            first.value : 0.0
                            second.value : 1.0

                            ToolTip {
                                parent : _slider.first.handle
                                visible : _slider.first.pressed
                                text : _slider.first.value.toFixed(1) + '℃'
                            }
                            ToolTip {
                                parent : _slider.second.handle
                                visible : _slider.second.pressed
                                text : _slider.second.value.toFixed(1) + '℃'
                            }
                        }

                        ColumnLayout {
                            Layout.fillHeight : true

                            Label {
                                text : max_temperature + '℃'
                            }
                            Rectangle {
                                Layout.fillHeight : true
                            }
                            Label {
                                text : min_temperature + '℃'
                            }
                        }
                    }

                    Button {
                        text : '설정'
                        Layout.fillWidth : true
                        Layout.alignment : Qt.AlignHCenter | Qt.AlignVCenter

                        onReleased : con.analysis_set_clim(_slider.first.value, _slider.second.value)
                    }
                }
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillWidth : true

            RowLayout {
                anchors.fill : parent

                ColumnLayout {
                    Layout.alignment : Qt.AlignLeft | Qt.AlignTop
                    Layout.fillHeight : true

                    Label {
                        font.weight : Font.Medium
                        text : '환경 변수'
                    }

                    GridLayout {
                        columns : 3

                        Label {
                            text : '실내 온도'
                        }
                        TextField {}
                        Label {
                            text : '℃'
                        }

                        Label {
                            text : '실외 온도'
                        }
                        TextField {}
                        Label {
                            text : '℃'
                        }
                    }
                }

                ColumnLayout {
                    Layout.fillHeight : true

                    Label {
                        font.weight : Font.Medium
                        text : '방사율 수정'
                    }

                    GridLayout {
                        columns : 2

                        Label {
                            text : '벽'
                        }
                        TextField {
                            id : _wall_emissivity
                            text : '0.90' // TODO 숫자 입력으로
                        }

                        Label {
                            text : '창문'
                        }
                        TextField {
                            id : _window_emissivity
                            text : '0.92'
                        }
                    }

                    Button {
                        Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                        text : '적용'

                        onReleased : {
                            let ewall = parseFloat(_wall_emissivity.text);
                            let ewindow = parseFloat(_window_emissivity.text);
                            con.analysis_correct_emissivity(ewall, ewindow);
                        }
                    }
                }

                ColumnLayout {
                    Layout.fillHeight : true

                    Label {
                        font.weight : Font.Medium
                        text : '지점 온도 보정'
                    }

                    GridLayout {
                        columns : 3

                        Label {
                            text : '열화상'
                        }
                        TextField {
                            id : _ir_temperature
                            readOnly : true
                        }
                        Label {
                            text : '℃'
                        }

                        Label {
                            text : '보정 온도'
                        }
                        TextField {
                            id : _reference_temperature
                            validator : DoubleValidator {}
                        }
                        Label {
                            text : '℃'
                        }
                    }

                    Button {
                        Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                        text : '보정'

                        onReleased : {
                            let temperature = parseFloat(_reference_temperature.text);
                            con.analysis_correct_temperature(temperature);
                        }
                    }
                }
            }
        }
    }

    function init() {
        con.analysis_plot()
    }

    function set_temperature_range(vmin, vmax) {
        min_temperature = vmin;
        max_temperature = vmax;
        _slider.first.value = vmin;
        _slider.second.value = vmax;
    }

    function show_point_temperature(value) {
        _ir_temperature.text = value
    }
}

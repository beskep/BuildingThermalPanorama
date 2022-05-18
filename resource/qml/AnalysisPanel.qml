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

            RowLayout {
                RowLayout {
                    ToolRadioButton {
                        id : _ir
                        text : '열화상'
                        checked : true
                    }
                    ToolRadioButton {
                        id : _factor
                        text : '지표'

                        onCheckedChanged : analysis_plot()
                    }
                }

                ToolSeparator {}

                CheckBox {
                    id : _segmentation

                    font.weight : Font.Medium
                    Material.accent : Material.color(Material.primaryColor, Material.Shade800)

                    text : '외피 부위 표시'

                    onReleased : analysis_plot()
                }
            }
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

                            enabled : _ir.checked

                            orientation : Qt.Vertical
                            from : min_temperature
                            to : max_temperature
                            stepSize : 0.1
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

                        enabled : _ir.checked
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
                spacing : 10

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

                            text : '0.90'
                            color : 'gray'

                            horizontalAlignment : TextInput.AlignRight
                            validator : DoubleValidator {}
                            onTextChanged : {
                                _ce_button.highlighted = true;
                                color = 'gray';
                            }
                        }

                        Label {
                            text : '창문'
                        }
                        TextField {
                            id : _window_emissivity

                            text : '0.92'
                            color : 'gray'

                            horizontalAlignment : TextInput.AlignRight
                            validator : DoubleValidator {}
                            onTextChanged : {
                                _ce_button.highlighted = true;
                                color = 'gray';
                            }
                        }
                    }

                    Button {
                        id : _ce_button
                        Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                        text : '적용'

                        onReleased : {
                            let ewall = parseFloat(_wall_emissivity.text);
                            let ewindow = parseFloat(_window_emissivity.text);
                            con.analysis_correct_emissivity(ewall, ewindow);
                            analysis_plot();

                            highlighted = false;
                            _wall_emissivity.color = 'black';
                            _window_emissivity.color = 'black';
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
                            horizontalAlignment : TextInput.AlignRight
                        }
                        Label {
                            text : '℃'
                        }

                        Label {
                            text : '보정 온도'
                        }
                        TextField {
                            id : _reference_temperature
                            horizontalAlignment : TextInput.AlignRight
                            validator : DoubleValidator {}
                            onTextChanged : {
                                _ct_button.highlighted = true;
                                color = 'gray';
                            }
                        }
                        Label {
                            text : '℃'
                        }
                    }

                    Button {
                        id : _ct_button

                        Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                        text : '보정'

                        onReleased : {
                            let temperature = parseFloat(_reference_temperature.text);
                            con.analysis_correct_temperature(temperature);
                            analysis_plot();

                            highlighted = false;
                            _reference_temperature.color = 'black';
                        }
                    }
                }

                ColumnLayout {
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
                        TextField {
                            id : _int_temperature

                            text : ''

                            horizontalAlignment : TextInput.AlignRight
                            validator : DoubleValidator {}
                            onTextChanged : {
                                _bt_button.highlighted = true;
                                color = 'gray'
                            }
                        }
                        Label {
                            text : '℃'
                        }

                        Label {
                            text : '실외 온도'
                        }
                        TextField {
                            id : _ext_temperature

                            text : ''

                            horizontalAlignment : TextInput.AlignRight
                            validator : DoubleValidator {}
                            onTextChanged : {
                                _bt_button.highlighted = true;
                                color = 'gray'
                            }
                        }
                        Label {
                            text : '℃'
                        }
                    }

                    Button {
                        id : _bt_button

                        Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                        text : '설정'

                        onReleased : {
                            let te = parseFloat(_ext_temperature.text);
                            let ti = parseFloat(_int_temperature.text);
                            con.analysis_set_teti(te, ti);

                            if (_factor.checked) {
                                analysis_plot()
                            } else {
                                _factor.checked = true;
                            }

                            highlighted = false;
                            _ext_temperature.color = 'black';
                            _int_temperature.color = 'black';
                        }
                    }
                }

                ColumnLayout {
                    spacing : 0

                    HorizontalHeaderView {
                        syncView : table_view
                        // Layout.fillWidth : true

                        model : ListModel {
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
                        delegate : Rectangle {
                            implicitHeight : 40
                            implicitWidth : 120
                            color : '#eeeeee'

                            Label {
                                text : name
                                anchors.centerIn : parent
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

                        model : TableModel {
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

                            rows : []
                        }

                        delegate : Rectangle {
                            implicitHeight : 40
                            implicitWidth : 120

                            Label {
                                text : display
                                anchors.centerIn : parent
                            }
                        }
                    }
                }
            }
        }
    }

    function analysis_plot() {
        con.analysis_plot(_factor.checked, _segmentation.checked)
    }

    function init() {
        analysis_plot()
    }

    function set_temperature_range(vmin, vmax) {
        min_temperature = vmin;
        max_temperature = vmax;
        _slider.first.value = vmin;
        _slider.second.value = vmax;

        if (!_ext_temperature.text.length) {
            _ext_temperature.text = vmin
        }

        if (!_int_temperature.text.length) {
            _int_temperature.text = vmax
        }
    }

    function show_point_temperature(value) {
        _ir_temperature.text = value
    }

    function clear_table() {
        table_model.clear()
    }

    function add_table_row(row) {
        table_model.appendRow(row)
    }
}

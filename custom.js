// --- custom4.js (Complete Updated File) ---
console.log("custom4.js script loaded and running.");

// --------------------------------------------
// Constants and Global State
// --------------------------------------------
const POLLUTANTS = [
    'Ozone',
    'Nitrogen dioxide',
    'Sulphur dioxide',
    'Particulate matter 10',
    'Particulate matter 2.5'
];
const STATIONS = [1, 2, 3, 4, 5, 7, 9,
     11, 12, 13, 14, 15, 17, 18, 19,
     21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 33, 34, 35, 37, 39];
const LOCAL_API_URL = 'http://localhost:8003/api';
const EAQI_LIMITS = {
    'NO2': [0, 40, 90, 120, 230, 340, 1000],
    'O3': [0, 50, 100, 130, 240, 380, 800],
    'PM10': [0, 20, 40, 50, 100, 150, 1200],
    'PM25': [0, 10, 20, 25, 50, 75, 800],
    'SO2': [0, 100, 200, 350, 500, 750, 1250]
};
const COLORS = ['#50f0e6', '#50ccaa', '#f0e641', '#ff5050', '#960032', '#7d2181'];
const CHART_COLORS = [
    'rgb(54, 162, 235)', 'rgb(255, 99, 132)', 'rgb(75, 192, 192)',
    'rgb(255, 206, 86)', 'rgb(153, 102, 255)', 'rgb(255, 159, 64)',
    'rgb(199, 199, 199)', 'rgb(83, 102, 255)', 'rgb(40, 159, 64)'
];

// Make chart instance and data globally accessible for easier debugging and updates
window.myApp = {
    chartData: null,
    myChart: null,
    clickedDistrictName: null,
    lastFeature: null, // To hold a stable reference to the last clicked feature
    updateStationChart: (feature) => updateStationChart(feature)
};

// --------------------------------------------
// API Call Functions
// --------------------------------------------

/**
 * Fetches historical and aggregated data for a specific station.
 * @param {number} stationId - The ID of the station.
 * @returns {Promise<object|null>} The station chart data or null on error.
 */
async function callStationChartApi(stationId) {
    const url = `${LOCAL_API_URL}/station/${stationId}`;
    console.log(`DEBUG: Calling Station Chart API: ${url}`);
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log(`DEBUG: API Response for Station ${stationId}:`, data);
        return data;
    } catch (error) {
        console.error(`Failed to fetch chart data for station ${stationId}:`, error);
        return null;
    }
}

/**
 * Fetches historical data for a building based on its coordinates.
 * @param {number} latitude - The latitude of the building.
 * @param {number} longitude - The longitude of the building.
 * @returns {Promise<object|null>} The building data or null on error.
 */
async function callBuildingApi(latitude, longitude) {
    const baseUrl = `${LOCAL_API_URL}/historical/building`;
    const params = new URLSearchParams({ lat: latitude, lon: longitude });
    const fullUrl = `${baseUrl}?${params.toString()}`;
    console.log(`DEBUG: Calling Building API: ${fullUrl}`);
    try {
        const response = await fetch(fullUrl);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log(`DEBUG: API Response for Building at ${latitude},${longitude}:`, data);
        return data;
    } catch (error) {
        console.error("Failed to fetch building data:", error);
        return null;
    }
}

/**
 * Fetches historical data for the Low Emission Zone.
 * @returns {Promise<object|null>} The LEZ data or null on error.
 */
async function callLezApi() {
    const url = `${LOCAL_API_URL}/historical/lez`;
    console.log(`DEBUG: Calling LEZ API: ${url}`);
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log(`DEBUG: API Response for LEZ:`, data);
        return data;
    } catch (error) {
        console.error("Failed to fetch LEZ data:", error);
        return null;
    }
}

/**
 * Fetches historical data for all districts.
 * @returns {Promise<object|null>} The districts data or null on error.
 */
async function callDistrictApi() {
    const url = `${LOCAL_API_URL}/historical/district`;
    console.log(`DEBUG: Calling District API: ${url}`);
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        const data = await response.json();
        console.log(`DEBUG: API Response for Districts:`, data);
        return data;
    } catch (error) {
        console.error("Failed to fetch District data:", error);
        return null;
    }
}


// --------------------------------------------
// Feature Processing & Data Loading
// --------------------------------------------

/**
 * Main handler for feature click events. Routes to the correct processor.
 * @param {object} feature - The clicked Terria feature.
 * @param {object} allStationsData - Pre-loaded data for all stations.
 */
async function handleFeatureClick(feature, allStationsData) {
    const catalogItemName = feature?._catalogItem?.uniqueId || '';
    const stationId = feature.properties?.station_id?.getValue();
    console.log("DEBUG: handleFeatureClick triggered for feature:", feature);

    if (stationId) {
        console.log(`DEBUG: Feature identified as Station ID: ${stationId}. Processing...`);
        await processStationFeature(feature, allStationsData);
    } else if (catalogItemName === "buildings") {
        console.log("DEBUG: Feature identified as Building. Processing...");
        await processBuildingFeature(feature);
    } else if (catalogItemName === "sofia_lez") {
        console.log("DEBUG: Feature identified as LEZ. Processing...");
        await processLowEmissionZoneFeature(feature);
    } else if (catalogItemName === "districts") {
        console.log("DEBUG: Feature identified as District. Processing...");
        await processDistrictFeature(feature);
    } else {
        console.log("Clicked on an unhandled feature type:", feature);
    }
}

/**
 * Processes a station feature on click.
 * @param {object} terriaFeature - The clicked station feature.
 * @param {object} allStationsData - Pre-loaded current data for all stations.
 */
async function processStationFeature(terriaFeature, allStationsData) {
    const stationId = terriaFeature.properties?.station_id?.getValue();
    if (!stationId) return;

    console.log(`DEBUG: processStationFeature for station ${stationId}`);
    if (allStationsData && allStationsData.stations[stationId]) {
        if (!terriaFeature.properties.airQualityData) {
            const stationData = allStationsData.stations[stationId];
            terriaFeature.properties.addProperty('airQualityData', stationData);
            terriaFeature.properties.addProperty('airQualityDate', allStationsData.date);
            console.log(`DEBUG: Added pre-loaded data to station feature ${stationId}.`);
        }
    } else {
         console.warn("processStationFeature: current station data not found for station:", stationId);
    }

    if (terriaFeature.properties.stationChartData || terriaFeature.properties.dataError) {
        console.log(`DEBUG: Chart data for station ${stationId} already loaded or errored. Skipping fetch.`);
        return;
    }
    const chartData = await callStationChartApi(stationId);
    if (chartData) {
        terriaFeature.properties.addProperty('stationChartData', chartData);
        console.log(`DEBUG: Successfully added chart data to station feature ${stationId}.`);
    } else {
        terriaFeature.properties.addProperty('dataError', 'Failed to load historical chart data.');
        console.error(`DEBUG: Failed to add chart data to station feature ${stationId}.`);
    }
}


async function processBuildingFeature(feature) {
    if (feature.properties.buildingChartData || feature.properties.dataError) return;
    const buildingLat = feature.properties?.Latitude?.getValue();
    const buildingLon = feature.properties?.Longitude?.getValue();
    if (buildingLat && buildingLon) {
        const buildingData = await callBuildingApi(buildingLat, buildingLon);
        if (buildingData) feature.properties.addProperty('buildingChartData', buildingData);
        else feature.properties.addProperty('dataError', 'Failed to load historical data.');
    }
}

async function processLowEmissionZoneFeature(feature) {
    console.log("DEBUG: Inside processLowEmissionZoneFeature. Checking for existing data...");
    if (feature.properties.lezChartData || feature.properties.dataError) {
        console.log("DEBUG: LEZ data already exists on feature. Skipping API call.", feature.properties);
        return;
    }
    const lezData = await callLezApi();
    if (lezData) feature.properties.addProperty('lezChartData', lezData);
    else feature.properties.addProperty('dataError', 'Failed to load historical data for LEZ.');
}

async function processDistrictFeature(feature) {
    if (feature.properties.districtChartData || feature.properties.dataError) return;
    const districtData = await callDistrictApi();
    if (districtData) feature.properties.addProperty('districtChartData', districtData);
    else feature.properties.addProperty('dataError', 'Failed to load historical data for Districts.');
}


// --------------------------------------------
// Charting Logic
// --------------------------------------------

/**
 * Main chart initializer. Routes to the specific chart handler based on feature type.
 * @param {object} feature - The feature for which to initialize the chart.
 */
function initializeChart(feature) {
    console.log("DEBUG: initializeChart called for feature:", feature);
    const stationId = feature.properties?.station_id?.getValue();

    if (stationId) {
        console.log(`DEBUG: Initializing chart for Station ${stationId}`);
        if (feature.properties?.stationChartData?.getValue()) {
            updateStationChart(feature);
        } else {
            console.log(`DEBUG: No stationChartData found on feature to initialize chart.`);
        }
    } else if (feature?._catalogItem?.uniqueId === "buildings") {
        if (feature.properties?.buildingChartData?.getValue()) {
            window.myApp.chartData = feature.properties.buildingChartData.getValue();
            updateBuildingChart();
        }
    } else if (feature?._catalogItem?.uniqueId === "sofia_lez") {
        if (feature.properties?.lezChartData?.getValue()) {
            window.myApp.chartData = feature.properties.lezChartData.getValue();
            updateLezChart();
        }
    } else if (feature?._catalogItem?.uniqueId === "districts") {
        window.myApp.clickedDistrictName = feature._properties._name_en.getValue();
        if (feature.properties?.districtChartData?.getValue()) {
            window.myApp.chartData = feature.properties.districtChartData.getValue();
            updateDistrictChart();
        }
    }
}

/**
 * Updates or creates the chart for the Station feature.
 * @param {object} feature - The feature object to create the chart for.
 */
function updateStationChart(feature) {
    console.log("DEBUG: updateStationChart function called.");
    if (!feature) {
        console.log("DEBUG: updateStationChart - No feature was passed to the function.");
        return;
    }

    const chartData = feature.properties?.stationChartData?.getValue();
    let measurement = document.getElementById('station-measurement-select')?.value;
    if (!measurement && chartData && chartData.last_24h && chartData.last_24h.length > 0) {
        measurement = chartData.last_24h[0].measurement;
        console.log(`DEBUG: Measurement was empty, defaulted to '${measurement}'`);
    }

    const compare = document.getElementById('station-compare-avg')?.checked;

    console.log(`DEBUG: updateStationChart - Measurement: ${measurement}, Compare: ${compare}`);
    if (!chartData || !measurement) {
        console.log("DEBUG: updateStationChart - Missing chartData or measurement. Bailing.", chartData);
        return;
    }

    const datasets = [];

    const last24hData = chartData.last_24h
        .filter(d => d.measurement === measurement)
        .map(d => ({ x: d.measured_time, y: d.reading_value }));

    datasets.push({
        label: `Last 24h (${measurement})`,
        data: last24hData,
        borderColor: CHART_COLORS[0],
        tension: 0.1,
        fill: false
    });

    if (compare) {
        const monthlyAvgData = chartData.monthly_avg_by_hour
            .filter(d => d.measurement === measurement);
        
        const mappedAvgData = last24hData.map(d => {
            const hour = new Date(d.x).getHours();
            const avgForHour = monthlyAvgData.find(avg => avg.hour_of_day === hour);
            return { x: d.x, y: avgForHour ? avgForHour.avg_reading : null };
        });

        datasets.push({
            label: `Monthly Avg (${measurement})`,
            data: mappedAvgData,
            borderColor: CHART_COLORS[1],
            borderDash: [5, 5],
            tension: 0.1,
            fill: false,
            spanGaps: true
        });
    }

    console.log("DEBUG: updateStationChart - Datasets prepared for rendering:", datasets);
    const timeOptions = { unit: 'hour', tooltipFormat: 'MMM d, HH:mm', displayFormats: { hour: 'HH:mm' } };
    renderChart('station-chart', datasets, 'time', { time: timeOptions });
}

/**
 * Updates or creates the chart for the Building feature.
 */
function updateBuildingChart() {
    const measurement = document.getElementById('measurement-select')?.value;
    const timePeriod = document.querySelector('input[name="time-period"]:checked')?.value;
    const chartData = window.myApp.chartData;
    if (!measurement || !timePeriod || !chartData) return;
    let dataToShow = [], label = '', timeOptions = {}, chartMappedData = [];
    if (timePeriod === 'daily') {
        dataToShow = chartData.daily_last_two_weeks.filter(d => d.measurement === measurement);
        label = `${measurement} (Daily)`;
        timeOptions.unit = 'day';
        chartMappedData = dataToShow.map(d => ({ x: d.modeled_date, y: d.reading_value }));
    } else {
        dataToShow = chartData.weekly_last_year.filter(d => d.measurement === measurement);
        label = `${measurement} (Weekly)`;
        timeOptions.unit = 'week';
        chartMappedData = dataToShow.map(d => ({ x: d.week_start, y: d.avg_reading_value }));
    }
    const datasets = [{ label: label, data: chartMappedData, borderColor: 'rgb(75, 192, 192)', tension: 0.1 }];
    renderChart('historical-chart', datasets, 'time', { time: timeOptions });
}

/**
 * Updates or creates the chart for the Low Emission Zone feature.
 */
function updateLezChart() {
    const measurement = document.getElementById('lez-measurement-select')?.value;
    const chartData = window.myApp.chartData;

    // ADDED: Console log to show the raw data being processed
    console.log("DEBUG LEZ Chart: Initial data from API and selected measurement", { measurement, chartData });

    if (!measurement || !chartData || chartData.length === 0) {
        // To prevent errors, render an empty chart if there's no data
        renderChart('lez-chart', [], 'category', {}, []);
        return;
    };

    // Filter data for the selected measurement
    const dataForMeasurement = chartData.filter(d => d.measurement === measurement);

    // Group data by "winter season" (e.g., Winter 2022-2023)
    const dataByYear = {};
    dataForMeasurement.forEach(d => {
        const date = new Date(d.week_start);
        const year = date.getMonth() >= 11 ? date.getFullYear() : date.getFullYear() - 1; // Dec is month 11
        if (!dataByYear[year]) dataByYear[year] = [];
        dataByYear[year].push(d);
    });

    // --- FIX: Dynamically generate labels from the data itself ---
    const monthOrder = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'];
    const labelSet = new Set();
    chartData.forEach(d => {
        const date = new Date(d.week_start);
        const label = `${date.toLocaleString('default', { month: 'short' })} ${date.getDate()}`;
        labelSet.add(label);
    });
    
    const labels = Array.from(labelSet).sort((a, b) => {
        const [monthA, dayA] = a.split(' ');
        const [monthB, dayB] = b.split(' ');
        const monthIndexA = monthOrder.indexOf(monthA);
        const monthIndexB = monthOrder.indexOf(monthB);
        if (monthIndexA !== monthIndexB) {
            return monthIndexA - monthIndexB;
        }
        return parseInt(dayA) - parseInt(dayB);
    });
    // --- END FIX ---

    const datasets = Object.keys(dataByYear).map((year, index) => {
        const yearData = dataByYear[year];
        const dataMap = new Map();
        yearData.forEach(d => {
            const date = new Date(d.week_start);
            const key = `${date.toLocaleString('default', { month: 'short' })} ${date.getDate()}`;
            dataMap.set(key, d.avg_value);
        });
        return {
            label: `Winter ${year}-${parseInt(year) + 1}`,
            data: labels.map(label => dataMap.get(label) || null),
            borderColor: CHART_COLORS[index % CHART_COLORS.length],
            tension: 0.1,
            fill: false,
            spanGaps: true
        };
    });

    console.log("DEBUG LEZ Chart: Final datasets for rendering", { labels, datasets });
    renderChart('lez-chart', datasets, 'category', {}, labels);
}

/**
 * Updates or creates the chart for the Districts feature.
 */
function updateDistrictChart() {
    const measurement = document.getElementById('district-measurement-select')?.value;
    const timePeriod = document.querySelector('input[name="district-time-period"]:checked')?.value;
    
    // --- MODIFIED BLOCK START ---
    const comparisonSelect = document.getElementById('district-comparison-select');
    // Get an array of all selected district names
    const comparisonDistricts = comparisonSelect ? Array.from(comparisonSelect.selectedOptions).map(option => option.value) : [];
    // --- MODIFIED BLOCK END ---

    const chartData = window.myApp.chartData;
    const clickedDistrict = window.myApp.clickedDistrictName;

    if (!measurement || !timePeriod || !chartData || !clickedDistrict) return;

    const districtDataKey = timePeriod === 'daily' ? 'daily_by_district' : 'weekly_by_district';
    const averageDataKey = timePeriod === 'daily' ? 'daily_overall_average' : 'weekly_overall_average';
    const dateKey = timePeriod === 'daily' ? 'modeled_date' : 'week_start';
    const valueKey = 'avg_value', avgValueKey = 'overall_avg_value';

    if (!chartData[districtDataKey] || !chartData[averageDataKey]) return;

    const datasets = [];

    // Dataset for the originally clicked district
    const clickedDistrictLower = clickedDistrict.toLowerCase();
    const clickedDistrictData = chartData[districtDataKey].filter(d => d.measurement === measurement && d.region_name.toLowerCase() === clickedDistrictLower).sort((a, b) => new Date(a[dateKey]) - new Date(b[dateKey])).map(d => ({ x: d[dateKey], y: d[valueKey] }));
    datasets.push({ label: clickedDistrict, data: clickedDistrictData, borderColor: CHART_COLORS[0], borderWidth: 2.5, tension: 0.1, fill: false });

    // Dataset for the city-wide average
    const overallAverageData = chartData[averageDataKey].filter(d => d.measurement === measurement).sort((a, b) => new Date(a[dateKey]) - new Date(b[dateKey])).map(d => ({ x: d[dateKey], y: d[avgValueKey] }));
    datasets.push({ label: 'City Average', data: overallAverageData, borderColor: CHART_COLORS[1], borderDash: [5, 5], tension: 0.1, fill: false });

    // --- MODIFIED BLOCK START ---
    // Loop through all selected comparison districts and create a dataset for each
    if (comparisonDistricts.length > 0) {
        comparisonDistricts.forEach((districtName, index) => {
            const districtNameLower = districtName.toLowerCase();
            const comparisonDistrictData = chartData[districtDataKey]
                .filter(d => d.measurement === measurement && d.region_name.toLowerCase() === districtNameLower)
                .sort((a, b) => new Date(a[dateKey]) - new Date(b[dateKey]))
                .map(d => ({ x: d[dateKey], y: d[valueKey] }));

            // Start color index at 2 to avoid conflicting with the primary lines
            const colorIndex = 2 + index;
            datasets.push({
                label: districtName,
                data: comparisonDistrictData,
                // Use modulo to cycle through colors if many districts are selected
                borderColor: CHART_COLORS[colorIndex % CHART_COLORS.length], 
                tension: 0.1,
                fill: false
            });
        });
    }
    // --- MODIFIED BLOCK END ---

    const timeOptions = { unit: timePeriod === 'daily' ? 'day' : 'week' };
    renderChart('district-chart', datasets, 'time', { time: timeOptions });
}


/**
 * Generic function to render a chart on a canvas.
 * @param {string} canvasId - The ID of the canvas element.
 * @param {Array<object>} datasets - The array of datasets for the chart.
 * @param {string} xScaleType - The type of the X-axis (e.g., 'time', 'category').
 * @param {object} xScaleOptions - Additional options for the X-axis scale.
 * @param {Array<string>} [labels] - Optional labels for the X-axis (used for category scale).
 */
function renderChart(canvasId, datasets, xScaleType, xScaleOptions, labels = []) {
    console.log(`DEBUG: renderChart called for canvas #${canvasId}`);
    if (typeof Chart === 'undefined' || typeof dateFns === 'undefined') {
        console.error('FATAL: Chart.js or date-fns is not loaded.');
        return;
    }
    const canvasElement = document.getElementById(canvasId);
    if (!canvasElement) {
        console.error(`DEBUG: renderChart - Canvas element #${canvasId} not found!`);
        return;
    }
    const ctx = canvasElement.getContext('2d');

    if (window.myApp.myChart) {
        window.myApp.myChart.destroy();
    }

    const chartConfig = {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: xScaleType,
                    adapters: { date: { locale: dateFns.enUS } },
                    ...xScaleOptions,
                    title: { display: true, text: 'Date / Time' }
                },
                y: { title: { display: true, text: 'Reading Value (µg/m³)' } }
            },
            interaction: { mode: 'index', intersect: false }
        }
    };
    
    if (xScaleType === 'category' && labels.length > 0) {
        chartConfig.data.labels = labels;
    }

    window.myApp.myChart = new Chart(ctx, chartConfig);
    console.log(`DEBUG: Chart successfully rendered on #${canvasId}.`);
}


// --------------------------------------------
// Popup HTML Generation
// --------------------------------------------

function generatePropertiesTableHtml(feature) {
    let html = '<table class="cesium-infoBox-defaultTable"><tbody>';
    const props = feature.properties;
    for (const key in props) {
        if (props.hasOwnProperty(key) && typeof props[key]?.getValue === 'function') {
            const value = props[key].getValue();
            if (typeof value === 'string' || typeof value === 'number') {
                const prettyKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `<tr><td style="padding:4px;">${prettyKey}</td><td style="padding:4px;">${value}</td></tr>`;
            }
        }
    }
    html += '</tbody></table>';
    return html;
}

function generatePopupHtml(feature) {
    console.log("DEBUG: generatePopupHtml called for feature:", feature);
    if (!feature || !feature.properties) return '<p>No data available for this feature.</p>';
    
    const stationId = feature.properties?.station_id?.getValue();
    if (stationId) {
        return generateStationPopupHtml(feature);
    } else if (feature?._catalogItem?.uniqueId === "buildings") {
        return generateBuildingPopupHtml(feature);
    } else if (feature?._catalogItem?.uniqueId === "sofia_lez") {
        return generateLowEmissionZonePopupHtml(feature);
    } else if (feature?._catalogItem?.uniqueId === "districts") {
        return generateDistrictPopupHtml(feature);
    }
    return '<h3>Unknown Feature</h3><p>No specific information available.</p>';
}

function generateStationPopupHtml(feature) {
    const props = feature.properties;
    const stationId = props.station_id?.getValue();
    const chartData = props.stationChartData?.getValue();
    const dataError = props.dataError?.getValue();
    console.log(`DEBUG: generateStationPopupHtml for station ${stationId}. Chart data available:`, !!chartData);

    let headerHtml = `<h3 style="margin:0 0 15px 0; padding-bottom:10px; border-bottom:1px solid #eee;">Station ${stationId}</h3>`;

    if (chartData && chartData.latest && chartData.latest.length > 0) {
        // --- CHANGED BLOCK START ---
        const latestDate = new Date(chartData.latest[0].measured_time);
        const formattedDate = latestDate.toLocaleString([], { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        headerHtml += `<h4>Latest Readings <span style="font-size: 0.8em; color: #666; font-weight: normal;">(${formattedDate})</span></h4>`;
        // --- CHANGED BLOCK END ---
        
        headerHtml += '<table class="cesium-infoBox-defaultTable"><tbody>';
        chartData.latest.forEach(reading => {
            headerHtml += `<tr><td style="padding:4px;">${reading.measurement}</td><td style="padding:4px;"><strong>${reading.reading_value.toFixed(2)}</strong> µg/m³</td></tr>`;
        });
        headerHtml += '</tbody></table><hr>';
    } 

    let chartHtml = '';
    if (dataError) {
        chartHtml = `<p style="color: red; margin-top: 15px;">${dataError}</p>`;
    } else if (chartData && chartData.last_24h && chartData.last_24h.length > 0) {
        const measurements = [...new Set(chartData.last_24h.map(item => item.measurement))].sort();
        let selectHtml = '<label for="station-measurement-select" style="margin-top:15px; display:block;">Select Measurement:</label>';
        
        // FIXED: Changed onchange to use the stable window.myApp.lastFeature reference
        selectHtml += `<select id="station-measurement-select" onchange="window.myApp.updateStationChart(window.myApp.lastFeature)" style="display: block; width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;">`;
        
        measurements.forEach((m, index) => {
            const isSelected = index === 0 ? 'selected' : '';
            selectHtml += `<option value="${m}" ${isSelected}>${m}</option>`;
        });
        selectHtml += '</select>';
        
        // FIXED: Changed onchange to use the stable window.myApp.lastFeature reference
        let comparisonHtml = `<div style="margin-top: 10px;"><label><input type="checkbox" id="station-compare-avg" onchange="window.myApp.updateStationChart(window.myApp.lastFeature)"> Compare to this month's daily average</label></div>`;
        
        chartHtml = `<h4>Last 24 Hours</h4>${selectHtml}${comparisonHtml}<div style="width:100%; height:300px; margin-top: 20px;"><canvas id="station-chart"></canvas></div>`;
        console.log("DEBUG: Chart HTML generated for station popup.");
    } else if (!chartData) {
        chartHtml = '<p>Loading chart data... <span class="loader"></span></p>';
    } else {
        chartHtml = '<p>No chart data available for the last 24 hours.</p>';
        console.log("DEBUG: No 24h data found in chartData object:", chartData);
    }

    return headerHtml + chartHtml;
}


function generateBuildingPopupHtml(feature) {
    const chartData = feature.properties?.buildingChartData?.getValue();
    const dataError = feature.properties?.dataError?.getValue();
    const propertiesTable = generatePropertiesTableHtml(feature);
    let chartHtml = '';
    if (dataError) {
        chartHtml = `<p style="color: red; margin-top: 15px;">${dataError}</p>`;
    } else if (chartData) {
        const allData = [...chartData.daily_last_two_weeks, ...chartData.weekly_last_year];
        const measurements = [...new Set(allData.map(item => item.measurement))];
        let selectHtml = '<label for="measurement-select" style="margin-top:15px; display:block;">Select Measurement:</label>';
        selectHtml += '<select id="measurement-select" onchange="updateBuildingChart()" style="display: block; width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;">';
        measurements.forEach(m => { selectHtml += `<option value="${m}">${m}</option>`; });
        selectHtml += '</select>';
        let timePeriodHtml = `<div style="margin-top: 10px; display: flex; gap: 15px;"><label><input type="radio" name="time-period" value="daily" onchange="updateBuildingChart()" checked> Daily</label><label><input type="radio" name="time-period" value="weekly" onchange="updateBuildingChart()"> Weekly</label></div>`;
        chartHtml = `${selectHtml}${timePeriodHtml}<div style="width:100%; height:300px; margin-top: 20px;"><canvas id="historical-chart"></canvas></div>`;
    }
    return `<h3>Building Details</h3>${propertiesTable}<hr>${chartHtml}`;
}

function generateLowEmissionZonePopupHtml(feature) {
    const chartData = feature.properties?.lezChartData?.getValue();
    const dataError = feature.properties?.dataError?.getValue();
    let chartHtml = '';
    if (dataError) {
        chartHtml = `<p style="color: red; margin-top: 15px;">${dataError}</p>`;
    } else if (chartData && chartData.length > 0) {
        const measurements = [...new Set(chartData.map(item => item.measurement))];
        let selectHtml = '<label for="lez-measurement-select" style="margin-top:15px; display:block;">Select Measurement:</label>';
        selectHtml += '<select id="lez-measurement-select" onchange="updateLezChart()" style="display: block; width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;">';
        measurements.forEach(m => { selectHtml += `<option value="${m}">${m}</option>`; });
        selectHtml += '</select>';
        chartHtml = `${selectHtml}<div style="width:100%; height:300px; margin-top: 20px;"><canvas id="lez-chart"></canvas></div>`;
    } else {
        chartHtml = '<p>No historical data available for the Low Emission Zone.</p>';
    }
    return `<h3>Low Emission Zone</h3>${generatePropertiesTableHtml(feature)}<hr>${chartHtml}`;
}

function generateDistrictPopupHtml(feature) {
    const chartData = feature.properties?.districtChartData?.getValue();
    const dataError = feature.properties?.dataError?.getValue();
    const districtName = feature.properties?.NAME_EN?.getValue() || feature.properties?.NAME?.getValue() || 'Districts';
    let chartHtml = '';

    if (dataError) {
        chartHtml = `<p style="color: red; margin-top: 15px;">${dataError}</p>`;
    } else if (chartData) {
        if (!chartData.daily_by_district || !chartData.daily_overall_average) {
             chartHtml = `<p style="color: orange; margin-top: 15px;">Waiting for correctly formatted data from the API...</p>`;
             return `<h3>${districtName}</h3>${generatePropertiesTableHtml(feature)}<hr><h4>${districtName} Historical Averages</h4>${chartHtml}`;
        }

        const allDistrictData = chartData.daily_by_district || [];
        const measurements = [...new Set(allDistrictData.map(item => item.measurement))];
        const allDistricts = [...new Set(allDistrictData.map(item => item.region_name))].sort();

        let selectHtml = '<label for="district-measurement-select" style="margin-top:15px; display:block;">Select Measurement:</label>';
        selectHtml += '<select id="district-measurement-select" onchange="updateDistrictChart()" style="display: block; width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;">';
        measurements.forEach(m => { selectHtml += `<option value="${m}">${m}</option>`; });
        selectHtml += '</select>';

        // --- MODIFIED BLOCK START ---
        let comparisonSelectHtml = '<label for="district-comparison-select" style="margin-top:10px; display:block;">Compare With (hold Ctrl/Cmd to select):</label>';
        comparisonSelectHtml += '<select id="district-comparison-select" onchange="updateDistrictChart()" multiple size="5" style="display: block; width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc;">';
        // Removed the "none" option, as it's implicit in a multi-select
        allDistricts.forEach(d => {
            if (d.toLowerCase() !== districtName.toLowerCase()) {
                comparisonSelectHtml += `<option value="${d}">${d}</option>`;
            }
        });
        comparisonSelectHtml += '</select>';
        // --- MODIFIED BLOCK END ---

        let timePeriodHtml = `<div style="margin-top: 10px; display: flex; gap: 15px;"><label><input type="radio" name="district-time-period" value="daily" onchange="updateDistrictChart()" checked> Daily</label><label><input type="radio" name="district-time-period" value="weekly" onchange="updateDistrictChart()"> Weekly</label></div>`;
        chartHtml = `${selectHtml}${comparisonSelectHtml}${timePeriodHtml}<div style="width:100%; height:300px; margin-top: 20px;"><canvas id="district-chart"></canvas></div>`;
    }

    return `<h3>${districtName}</h3>${generatePropertiesTableHtml(feature)}<hr><h4>${districtName} Historical Averages</h4>${chartHtml}`;
}


// --------------------------------------------
// Main Application & Interaction Logic
// --------------------------------------------

function hideTerriaInfo() {
    setTimeout(() => {
        const defaultCloseButton = document.querySelector('.tjs-feature-info-panel__btn--close-feature');
        if (defaultCloseButton) defaultCloseButton.click();
    }, 10);
}

function setupStaticPopupInteraction(allStationsData) {
    const popupStyles = `
        #static-popup { position: fixed; top: 55px; right: 15px; width: 400px; max-width: 90vw; min-width: 400px; /* ADDED min-width */ height: calc(100vh - 70px); background-color: white; border-left: 1px solid #ccc; box-shadow: -2px 0 5px rgba(0,0,0,0.1); transform: translateX(110%); transition: transform 0.3s ease-in-out; z-index: 1000; display: flex; flex-direction: column; border-radius: 8px; }
        #static-popup.is-visible { transform: translateX(0); }
        #popup-header { display: flex; justify-content: flex-end; padding: 10px; border-bottom: 1px solid #eee; }
        #popup-close-button { background: none; border: none; font-size: 24px; cursor: pointer; line-height: 1; }
        #popup-content { padding: 20px; overflow-y: auto; flex-grow: 1; }
        #popup-resizer {
            position: absolute;
            left: 0;
            top: 0;
            width: 10px;
            height: 100%;
            cursor: ew-resize; /* East-West resize cursor */
            z-index: 1001; /* Make sure it's on top */
        }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 2s linear infinite; display: inline-block; margin-left: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    `;
    const styleElement = document.createElement('style');
    styleElement.innerHTML = popupStyles;
    document.head.appendChild(styleElement);

    const existingPopup = document.getElementById('static-popup');
    if (existingPopup) {
        existingPopup.remove();
    }

    // 2. Always create the new popup with the correct structure
    const popupHtml = `<div id="static-popup"><div id="popup-resizer"></div><div id="popup-header"><button id="popup-close-button">&times;</button></div><div id="popup-content"></div></div>`;
    document.body.insertAdjacentHTML('beforeend', popupHtml);

    const terria = window.viewState.terria;
    const popup = document.getElementById('static-popup');
    const popupContent = document.getElementById('popup-content');
    const closeButton = document.getElementById('popup-close-button');
    const hidePopup = () => popup.classList.remove('is-visible');
    const showPopup = () => popup.classList.add('is-visible');
    closeButton.addEventListener('click', hidePopup);

    const resizer = document.getElementById('popup-resizer');
    
    const resize = (e) => {
        // Calculate the new width based on the mouse's horizontal position
        // window.innerWidth - e.clientX gives the distance from the right edge of the screen
        const newWidth = window.innerWidth - e.clientX;
        
        // Apply constraints
        if (newWidth >= 400 && newWidth <= 1000) {
            popup.style.width = `${newWidth}px`;
        }
    };

    resizer.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent text selection while dragging
        // Add listeners to the whole document to track mouse movement anywhere on the page
        document.addEventListener('mousemove', resize, false);
        document.addEventListener('mouseup', () => {
            // Clean up the listeners when the mouse button is released
            document.removeEventListener('mousemove', resize, false);
        }, { once: true }); // The mouseup listener only needs to fire once
    });

    let lastProcessedFeature = null;
    setInterval(async () => {
        const currentFeature = terria.pickedFeatures?.features[0];

        if (currentFeature && currentFeature !== lastProcessedFeature) {
            lastProcessedFeature = currentFeature;
            // FIXED: Store a stable reference to the feature for onchange events
            window.myApp.lastFeature = currentFeature;
            console.log("DEBUG: New feature picked:", currentFeature);

            hideTerriaInfo();
            popupContent.innerHTML = generatePopupHtml(currentFeature);
            showPopup();

            await handleFeatureClick(currentFeature, allStationsData);

            if (lastProcessedFeature === currentFeature && popup.classList.contains('is-visible')) {
                console.log("DEBUG: Re-rendering popup content after data fetch.");
                popupContent.innerHTML = generatePopupHtml(currentFeature);
                initializeChart(currentFeature);
            }
        }
    }, 250);
}

function calculateEAQI(stationData) {
    let maxBin = 0;
    for (const pollutantName in stationData.pollutants) {
        const value = stationData.pollutants[pollutantName];
        if (value !== null && EAQI_LIMITS[pollutantName]) {
            const limits = EAQI_LIMITS[pollutantName];
            for (let i = 1; i < limits.length; i++) {
                if (value < limits[i]) {
                    if (i > maxBin) maxBin = i;
                    break;
                }
            }
        }
    }
    return maxBin > 0 ? maxBin : null;
}

async function fetchAndProcessData() {
    console.log("DEBUG: fetchAndProcessData - Fetching latest station data for styling.");
    try {
        const response = await fetch(`${LOCAL_API_URL}/latest-station-data`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const dbRows = await response.json();
        console.log("DEBUG: API Response for latest-station-data:", dbRows);
        if (dbRows.length === 0) {
            console.warn("No data returned from the database for styling.");
            return null;
        }
        const aggregatedData = { stations: {}, date: new Date(dbRows[0].measured_time).toISOString().slice(0, 16).replace('T', ' ') };
        for (const station of STATIONS) aggregatedData.stations[station] = { pollutants: {}, eaqi: null };
        for (const row of dbRows) {
            if (aggregatedData.stations[row.station_id]) {
                aggregatedData.stations[row.station_id].pollutants[row.measurement] = row.reading_value;
            }
        }
        for (const stationId in aggregatedData.stations) {
            aggregatedData.stations[stationId].eaqi = calculateEAQI(aggregatedData.stations[stationId]);
        }
        console.log("DEBUG: Aggregated station data for styling:", aggregatedData);
        return aggregatedData;
    } catch (error) {
        console.error("Error fetching or processing styling data:", error);
        return null;
    }
}

function applyStyling(catalogItem, allStationsData) {
  console.log(`DEBUG: Applying styling to catalog item "${catalogItem.name}"`);
  const getEaqiColor = (eaqi) => (eaqi === null || eaqi === undefined) ? 'gray' : COLORS[eaqi - 1] || 'white';
  const styleConditions = [];
  const opacity = 0.7;

  for (const stationName in allStationsData.stations) {
    const color = getEaqiColor(allStationsData.stations[stationName].eaqi);
    styleConditions.push([`String(\${feature['station_id']}) === '${stationName}'`, `color('${color}', ${opacity})`]);
  }
  styleConditions.push(['true', `color('white', ${opacity})`]);
  
  console.log("DEBUG: Generated style conditions:", styleConditions);
  catalogItem.setTrait('user','style',{ color: { conditions: styleConditions } });
  catalogItem.loadMapItems();
  console.log("DEBUG: Styling applied and map items reloaded.");
}

async function add_current_station_data() {
    const terria = window.viewState.terria;
    const allStationsData = await fetchAndProcessData();
    if (!allStationsData) {
        console.error("DEBUG: Halting styling because fetchAndProcessData returned null.");
        return null;
    }

    const processedItems = new Set();
    const intervalId = setInterval(() => {
        const itemsToProcess = terria.workbench.items.filter(item => item.name === "Station Beams" && !processedItems.has(item.uniqueId));
        if (itemsToProcess.length > 0) {
            console.log(`DEBUG: Found ${itemsToProcess.length} "Station Beams" layer(s) to style.`);
            itemsToProcess.forEach(item => {
                applyStyling(item, allStationsData);
                processedItems.add(item.uniqueId);
            });
             clearInterval(intervalId);
        }
    }, 1000);
    return allStationsData;
}

async function onTerriaReady() {
    console.log("Waiting for Terria to be ready (8 seconds)...");
    setTimeout(async () => {
        console.log("Terria ready. Initializing custom script.");
        const allStationsData = await add_current_station_data();
        if (allStationsData) {
            console.log("DEBUG: Station styling data loaded, setting up popup interaction.");
            setupStaticPopupInteraction(allStationsData);
        } else {
            console.error("Could not set up popup because station styling data failed to load.");
        }
    }, 8000);
}

// Start the application
onTerriaReady();
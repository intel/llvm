// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state
let activeRuns = new Set(defaultCompareNames);
let chartInstances = new Map();
let timeseriesData, barChartsData, allRunNames;

// DOM Elements
let runSelect, selectedRunsDiv, suiteFiltersContainer;

// Run selector functions
function updateSelectedRuns() {
    selectedRunsDiv.innerHTML = '';
    activeRuns.forEach(name => {
        selectedRunsDiv.appendChild(createRunElement(name));
    });
    updateCharts();
}

function createRunElement(name) {
    const runElement = document.createElement('span');
    runElement.className = 'selected-run';
    runElement.innerHTML = `${name} <button onclick="removeRun('${name}')">X</button>`;
    return runElement;
}

function addSelectedRun() {
    const selectedRun = runSelect.value;
    if (selectedRun && !activeRuns.has(selectedRun)) {
        activeRuns.add(selectedRun);
        updateSelectedRuns();
    }
}

function removeRun(name) {
    activeRuns.delete(name);
    updateSelectedRuns();
}

// Chart creation and update
function createChart(data, containerId, type) {
    if (chartInstances.has(containerId)) {
        chartInstances.get(containerId).destroy();
    }

    const ctx = document.getElementById(containerId).getContext('2d');
    const options = {
        responsive: true,
        plugins: {
            title: { display: true, text: data.label },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        if (type === 'time') {
                            const point = context.raw;
                            return [
                                `${context.dataset.label}:`,
                                `Value: ${point.y.toFixed(2)} ${data.unit}`,
                                `Stddev: ${point.stddev.toFixed(2)} ${data.unit}`,
                                `Git Hash: ${point.gitHash}`,
                            ];
                        } else {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${data.unit}`;
                        }
                    }
                }
            }
        },
        scales: {
            y: {
                title: { display: true, text: data.unit }
            }
        }
    };

    if (type === 'time') {
        options.interaction = { mode: 'nearest', intersect: false };
        options.onClick = (event, elements) => {
            if (elements.length > 0) {
                const point = elements[0].element.$context.raw;
                if (point.gitHash && point.gitRepo) {
                    window.open(`https://github.com/${point.gitRepo}/commit/${point.gitHash}`, '_blank');
                }
            }
        };
        options.scales.x = {
            type: 'time',
            time: {
                displayFormats: { datetime: 'MMM d, yyyy HH:mm' },
                stepSize: 12
            },
            ticks: {
                maxRotation: 45,
                minRotation: 45,
                autoSkip: true,
                maxTicksLimit: 10
            }
        };
    }

    const chartConfig = {
        type: type === 'time' ? 'line' : 'bar',
        data: type === 'time' 
            ? { datasets: createTimeseriesDatasets(data) } 
            : { labels: data.labels, datasets: data.datasets },
        options: options
    };

    const chart = new Chart(ctx, chartConfig);
    chartInstances.set(containerId, chart);
    return chart;
}

function createTimeseriesDatasets(data) {
    return Object.entries(data.runs).map(([name, points]) => ({
        label: name,
        data: points.map(p => ({
            x: new Date(p.date),
            y: p.value,
            gitHash: p.git_hash,
            gitRepo: p.github_repo,
            stddev: p.stddev
        })),
        borderWidth: 1,
        pointRadius: 3,
        pointStyle: 'circle',
        pointHoverRadius: 5
    }));
}

function updateCharts() {
    // Filter data by active runs
    const filteredTimeseriesData = timeseriesData.map(chart => ({
        ...chart,
        runs: Object.fromEntries(
            Object.entries(chart.runs).filter(([name]) => activeRuns.has(name))
        )
    }));
    
    const filteredBarChartsData = barChartsData.map(chart => ({
        ...chart,
        labels: chart.labels.filter(label => activeRuns.has(label)),
        datasets: chart.datasets.map(dataset => ({
            ...dataset,
            data: dataset.data.filter((_, i) => activeRuns.has(chart.labels[i]))
        }))
    }));

    // Draw charts with filtered data
    drawCharts(filteredTimeseriesData, filteredBarChartsData);
}

function drawCharts(filteredTimeseriesData, filteredBarChartsData) {
    // Clear existing charts
    document.querySelectorAll('.charts').forEach(container => container.innerHTML = '');
    chartInstances.forEach(chart => chart.destroy());
    chartInstances.clear();
    
    // Create timeseries charts
    filteredTimeseriesData.forEach((data, index) => {
        const containerId = `timeseries-${index}`;
        const container = createChartContainer(data, containerId);
        document.querySelector('.timeseries .charts').appendChild(container);
        createChart(data, containerId, 'time');
    });

    // Create bar charts
    filteredBarChartsData.forEach((data, index) => {
        const containerId = `barchart-${index}`;
        const container = createChartContainer(data, containerId);
        document.querySelector('.bar-charts .charts').appendChild(container);
        createChart(data, containerId, 'bar');
    });

    // Apply current filters
    filterCharts();
}

function createChartContainer(data, canvasId) {
    const container = document.createElement('div');
    container.className = 'chart-container';
    container.setAttribute('data-label', data.label);
    container.setAttribute('data-suite', data.suite);
    
    const canvas = document.createElement('canvas');
    canvas.id = canvasId;
    container.appendChild(canvas);
    
    return container;
}

// URL and filtering functions
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

function updateURL() {
    const url = new URL(window.location);
    const regex = document.getElementById('bench-filter').value;
    const activeSuites = getActiveSuites();

    if (regex) {
        url.searchParams.set('regex', regex);
    } else {
        url.searchParams.delete('regex');
    }

    if (activeSuites.length > 0) {
        url.searchParams.set('suites', activeSuites.join(','));
    } else {
        url.searchParams.delete('suites');
    }

    history.replaceState(null, '', url);
}

function filterCharts() {
    const regexInput = document.getElementById('bench-filter').value;
    const regex = new RegExp(regexInput, 'i');
    const activeSuites = getActiveSuites();
    
    document.querySelectorAll('.chart-container').forEach(container => {
        const label = container.getAttribute('data-label');
        const suite = container.getAttribute('data-suite');
        container.style.display = (regex.test(label) && activeSuites.includes(suite)) ? '' : 'none';
    });

    updateURL();
}

function getActiveSuites() {
    return Array.from(document.querySelectorAll('.suite-checkbox:checked'))
        .map(checkbox => checkbox.getAttribute('data-suite'));
}

// Data processing
function processTimeseriesData(benchmarkRuns) {
    const resultsByLabel = {};
    
    benchmarkRuns.forEach(run => {
        const runDate = run.date ? new Date(run.date) : null;
        console.log(run);
        run.results.forEach(result => {
            if (!resultsByLabel[result.label]) {
                resultsByLabel[result.label] = {
                    label: result.label,
                    suite: result.suite,
                    unit: result.unit,
                    runs: {}
                };
            }
            
            if (!resultsByLabel[result.label].runs[run.name]) {
                resultsByLabel[result.label].runs[run.name] = [];
            }
            
            resultsByLabel[result.label].runs[run.name].push({
                date: runDate,
                value: result.value,
                stddev: result.stddev,
                git_hash: run.git_hash,
                github_repo: run.github_repo
            });
        });
    });
    
    return Object.values(resultsByLabel);
}

function processBarChartsData(benchmarkRuns) {
    const groupedResults = {};
    
    benchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            if (!result.explicit_group) return;
            
            if (!groupedResults[result.explicit_group]) {
                groupedResults[result.explicit_group] = {
                    label: result.explicit_group,
                    suite: result.suite,
                    unit: result.unit,
                    labels: [],
                    datasets: []
                };
            }
            
            const group = groupedResults[result.explicit_group];
            
            if (!group.labels.includes(run.name)) {
                group.labels.push(run.name);
            }
            
            let dataset = group.datasets.find(d => d.label === result.label);
            if (!dataset) {
                dataset = {
                    label: result.label,
                    data: new Array(group.labels.length).fill(null)
                };
                group.datasets.push(dataset);
            }
            
            const runIndex = group.labels.indexOf(run.name);
            dataset.data[runIndex] = result.value;
        });
    });
    
    return Object.values(groupedResults);
}

// Setup functions
function setupRunSelector() {
    runSelect = document.getElementById('run-select');
    selectedRunsDiv = document.getElementById('selected-runs');

    allRunNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        runSelect.appendChild(option);
    });

    updateSelectedRuns();
}

function setupSuiteFilters() {
    suiteFiltersContainer = document.getElementById('suite-filters');

    suiteNames.forEach(suite => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'suite-checkbox';
        checkbox.dataset.suite = suite;
        checkbox.checked = true;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + suite));
        suiteFiltersContainer.appendChild(label);
        suiteFiltersContainer.appendChild(document.createTextNode(' '));
    });
}

function initializeCharts() {
    // Process raw data
    timeseriesData = processTimeseriesData(benchmarkRuns);
    barChartsData = processBarChartsData(benchmarkRuns);
    allRunNames = [...new Set(benchmarkRuns.map(run => run.name))];
    
    // Setup UI components
    setupRunSelector();
    setupSuiteFilters();

    // Apply URL parameters
    const regexParam = getQueryParam('regex');
    const suitesParam = getQueryParam('suites');

    if (regexParam) {
        document.getElementById('bench-filter').value = regexParam;
    }

    if (suitesParam) {
        const suites = suitesParam.split(',');
        document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
            checkbox.checked = suites.includes(checkbox.getAttribute('data-suite'));
        });
    }

    // Setup event listeners
    document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', filterCharts);
    });
    document.getElementById('bench-filter').addEventListener('input', filterCharts);

    // Draw initial charts
    updateCharts();
}

// Make functions available globally for onclick handlers
window.addSelectedRun = addSelectedRun;
window.removeRun = removeRun;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeCharts);

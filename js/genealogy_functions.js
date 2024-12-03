// Global variables
const BOX_WIDTH = 180;
const MIN_SCALE = 0.05;
const MAX_SCALE = 1;
let currentFocus = 'var_micah';
let currentScale = 0.5;
let currentSuggestionIndex = -1;
let suggestions = [];
const container = document.getElementById('treeContainer');

// dealing with FOUC
// function ensureVisibility() {
//     document.documentElement.style.visibility = 'visible';
//     document.documentElement.style.opacity = '1';
// }
function ensureVisibility() {
    document.documentElement.classList.add('hide-fouc');
}

// Mobile detection
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth <= 768;
}

function handleMobileAccess() {
    if (isMobileDevice()) {
        const desktopContent = document.querySelector('.desktop-content');
        const mobileWarning = document.querySelector('.mobile-warning');
        const zoomControls = document.querySelector('.zoom-controls');
        
        // Clean up any existing visualization elements
        if (zoomControls) {
            zoomControls.remove();
        }
        if (container) {
            container.innerHTML = '';
        }
        
        if (desktopContent && mobileWarning) {
            desktopContent.style.display = 'none';
            mobileWarning.style.display = 'block';
        }
        
        return true;
    }
    return false;
}

// Add mobile check to the initialization
function loadFamilyData() {
    if (isMobileDevice()) {
        handleMobileAccess();
        return; // Don't load family data on mobile devices
    }

    Promise.all([
        fetch('../../data/name_box.json'),
        fetch('../../data/parent_mapping.json')
    ])
        .then(responses => Promise.all(responses.map(response => response.json())))
        .then(([familyData, parentMapping]) => {
            console.log('Data loaded:', { familyData, parentMapping });
            initializeVisualization(familyData, parentMapping);
        })
        .catch(error => console.error('Error loading data:', error));
}

function createZoomControls() {
    const zoomBar = document.createElement('div');
    zoomBar.className = 'zoom-controls';
    zoomBar.innerHTML = `
        <button id="zoomIn" class="zoom-btn">+</button>
        <input type="range" id="zoomSlider" min="5" max="100" value="50" orient="vertical">
        <button id="zoomOut" class="zoom-btn">-</button>
    `;
    document.body.appendChild(zoomBar);

    document.getElementById('zoomIn').addEventListener('click', () => handleZoom(0.1));
    document.getElementById('zoomOut').addEventListener('click', () => handleZoom(-0.1));
    document.getElementById('zoomSlider').addEventListener('input', (e) => {
        currentScale = e.target.value / 100;
        centerOnFocus(window.familyData, window.totalHeight);
    });
}

function handleZoom(delta) {
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, currentScale + delta));
    if (newScale !== currentScale) {
        currentScale = newScale;
        document.getElementById('zoomSlider').value = currentScale * 100;
        centerOnFocus(window.familyData, window.totalHeight);
    }
}

function createFamilyMemberBox(id, member, totalHeight) {
    const box = document.createElement('div');
    box.className = `person-box${id === currentFocus ? ' focused' : ''}`;
    box.setAttribute('data-id', id);
    
    const y = totalHeight - member.y - member.box_height;
    
    box.style.left = `${member.x}px`;
    box.style.top = `${y}px`;
    box.style.height = `${member.box_height}px`;
    box.style.width = `${BOX_WIDTH}px`;
    
    box.innerHTML = `
        <h3>${member.name}</h3>
        ${member.birth && member.birth !== '<NA>' ? `<p>b. ${member.birth}</p>` : ''}
    `;
    
    return box;
}

function createFamilyConnections(parentMapping, familyData, totalHeight) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.pointerEvents = 'none';
    
    const BOX_PADDING = 15;
    
    Object.values(parentMapping).forEach(chunk => {
        const parents = chunk.parent_variable_names.map(name => familyData[name]);
        const children = chunk.children_variable_names.map(name => familyData[name]);

        if (parents.length === 2) {
            const parent1BoxY = totalHeight - parents[0].y - parents[0].box_height;
            const parent2BoxY = totalHeight - parents[1].y - parents[1].box_height;
            
            const parent1Y = parent1BoxY + parents[0].box_height + (BOX_PADDING * 2);
            const parent2Y = parent2BoxY + parents[1].box_height + (BOX_PADDING * 2);
            
            const parent1X = parents[0].x + BOX_WIDTH/2;
            const parent2X = parents[1].x + BOX_WIDTH/2;
            
            const childrenY = totalHeight - (children[0].y + children[0].box_height);
            
            const parentMidX = (parent1X + parent2X) / 2;
            const childrenMidX = (children[0].x + children[children.length - 1].x + BOX_WIDTH) / 2;
            const jointX = (parentMidX + childrenMidX) / 2;
            const jointY = (parent1Y + childrenY) / 2;

            [parent1X, parent2X].forEach((parentX, idx) => {
                const parentY = idx === 0 ? parent1Y : parent2Y;
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("d", `
                    M ${parentX} ${parentY}
                    L ${parentX} ${jointY}
                    L ${jointX} ${jointY}
                `);
                path.setAttribute("class", "family-line");
                svg.appendChild(path);
            });

            children.forEach(child => {
                const childX = child.x + BOX_WIDTH/2;
                const childY = totalHeight - (child.y + child.box_height);
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("d", `
                    M ${jointX} ${jointY}
                    L ${childX} ${jointY}
                    L ${childX} ${childY}
                `);
                path.setAttribute("class", "family-line");
                svg.appendChild(path);
            });
        }
    });
    
    return svg;
}

function clearNuclearFamilyHighlights() {
    document.querySelectorAll('.nuclear-family').forEach(box => {
        box.classList.remove('nuclear-family', 'spouse', 'child');
    });
}

// function highlightNuclearFamily(familyData, parentMapping, focusedId) {
//     clearNuclearFamilyHighlights();
    
//     const familyUnit = Object.values(parentMapping).find(family => 
//         family.parent_variable_names.includes(focusedId)
//     );
    
//     if (familyUnit) {
//         const spouseId = familyUnit.parent_variable_names.find(id => id !== focusedId);
        
//         if (spouseId) {
//             const spouseBox = document.querySelector(`[data-id="${spouseId}"]`);
//             if (spouseBox) {
//                 spouseBox.classList.add('nuclear-family', 'spouse');
//             }
//         }
        
//         familyUnit.children_variable_names.forEach(childId => {
//             const childBox = document.querySelector(`[data-id="${childId}"]`);
//             if (childBox) {
//                 childBox.classList.add('nuclear-family', 'child');
//             }
//         });
//     }
// }

function highlightNuclearFamily(familyData, parentMapping, focusedId) {
    clearNuclearFamilyHighlights();
    
    // Find if person is a parent (existing code)
    const familyUnitAsParent = Object.values(parentMapping).find(family => 
        family.parent_variable_names.includes(focusedId)
    );
    
    // Find if person is a child (new code)
    const familyUnitAsChild = Object.values(parentMapping).find(family => 
        family.children_variable_names.includes(focusedId)
    );
    
    // Highlight person's spouse and children (if they are a parent)
    if (familyUnitAsParent) {
        const spouseId = familyUnitAsParent.parent_variable_names.find(id => id !== focusedId);
        
        if (spouseId) {
            const spouseBox = document.querySelector(`[data-id="${spouseId}"]`);
            if (spouseBox) {
                spouseBox.classList.add('nuclear-family', 'spouse');
            }
        }
        
        familyUnitAsParent.children_variable_names.forEach(childId => {
            const childBox = document.querySelector(`[data-id="${childId}"]`);
            if (childBox) {
                childBox.classList.add('nuclear-family', 'child');
            }
        });
    }
    
    // Highlight person's parents (if they are a child)
    if (familyUnitAsChild) {
        familyUnitAsChild.parent_variable_names.forEach(parentId => {
            const parentBox = document.querySelector(`[data-id="${parentId}"]`);
            if (parentBox) {
                parentBox.classList.add('nuclear-family', 'parent');
            }
        });
    }
}

function renderFamilyTree(familyData, totalHeight) {
    const existingBoxes = container.querySelectorAll('.person-box');
    existingBoxes.forEach(box => box.remove());
    
    Object.entries(familyData).forEach(([id, member]) => {
        const box = createFamilyMemberBox(id, member, totalHeight);
        container.appendChild(box);
    });
}

function centerOnFocus(familyData, totalHeight) {
    const focusedMember = familyData[currentFocus];
    if (!focusedMember) return;
    
    const viewportCenterX = window.innerWidth/2;
    const viewportCenterY = window.innerHeight/2;
    
    const memberCenterX = focusedMember.x + BOX_WIDTH/2;
    const memberCenterY = totalHeight - focusedMember.y - focusedMember.box_height/2;
    
    const dx = viewportCenterX - (memberCenterX * currentScale);
    const dy = viewportCenterY - (memberCenterY * currentScale);
    
    container.style.transformOrigin = '0 0';
    container.style.transform = `matrix(${currentScale}, 0, 0, ${currentScale}, ${dx}, ${dy})`;
}

function setupSearch() {
    const searchInput = document.getElementById('searchInput');
    const suggestionsContainer = document.getElementById('searchSuggestions');
    
    searchInput.addEventListener('input', handleSearch);
    searchInput.addEventListener('keydown', handleSearchKeydown);
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
            suggestionsContainer.style.display = 'none';
        }
    });
}

function handleSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    const suggestionsContainer = document.getElementById('searchSuggestions');
    
    if (!searchTerm) {
        suggestionsContainer.style.display = 'none';
        suggestions = [];
        currentSuggestionIndex = -1;
        return;
    }
    
    suggestions = Object.entries(window.familyData)
        .filter(([id, member]) => {
            return member.name.toLowerCase().includes(searchTerm) ||
                   id.toLowerCase().includes(searchTerm);
        })
        .slice(0, 5);
    
    if (suggestions.length > 0) {
        suggestionsContainer.innerHTML = suggestions
            .map(([id, member], index) => `
                <div class="suggestion-item" data-id="${id}">
                    ${member.name}
                </div>
            `)
            .join('');
            
        suggestionsContainer.style.display = 'block';
        
        document.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', () => selectSuggestion(item.dataset.id));
        });
        
        currentSuggestionIndex = -1;
    } else {
        suggestionsContainer.innerHTML = '<div class="suggestion-item">No results found</div>';
        suggestionsContainer.style.display = 'block';
    }
}

function handleSearchKeydown(e) {
    const suggestionsContainer = document.getElementById('searchSuggestions');
    
    if (!suggestions.length) return;
    
    const suggestionElements = document.querySelectorAll('.suggestion-item');
    
    switch(e.key) {
        case 'ArrowDown':
            e.preventDefault();
            currentSuggestionIndex = Math.min(currentSuggestionIndex + 1, suggestions.length - 1);
            updateSelectedSuggestion(suggestionElements);
            break;
            
        case 'ArrowUp':
            e.preventDefault();
            currentSuggestionIndex = Math.max(currentSuggestionIndex - 1, 0);
            updateSelectedSuggestion(suggestionElements);
            break;
            
        case 'Enter':
            e.preventDefault();
            if (currentSuggestionIndex >= 0) {
                selectSuggestion(suggestions[currentSuggestionIndex][0]);
            } else if (suggestions.length > 0) {
                selectSuggestion(suggestions[0][0]);
            }
            break;
            
        case 'Escape':
            suggestionsContainer.style.display = 'none';
            break;
    }
}

function updateSelectedSuggestion(elements) {
    elements.forEach(el => el.classList.remove('selected'));
    if (currentSuggestionIndex >= 0) {
        elements[currentSuggestionIndex].classList.add('selected');
        elements[currentSuggestionIndex].scrollIntoView({ block: 'nearest' });
    }
}

function selectSuggestion(id) {
    const searchInput = document.getElementById('searchInput');
    const suggestionsContainer = document.getElementById('searchSuggestions');
    
    currentFocus = id;
    renderFamilyTree(window.familyData, window.totalHeight);
    centerOnFocus(window.familyData, window.totalHeight);
    highlightNuclearFamily(window.familyData, window.parentMapping, currentFocus);
    
    searchInput.value = '';
    suggestionsContainer.style.display = 'none';
    suggestions = [];
    currentSuggestionIndex = -1;
}

function setupNavigation(familyData, parentMapping, totalHeight) {
    document.addEventListener('keydown', (e) => {
        const currentMember = familyData[currentFocus];
        if (!currentMember) return;
        
        let nextId = null;
        switch(e.key) {
            case 'ArrowUp':
                e.preventDefault();
                nextId = currentMember.arrows.up;
                break;
            case 'ArrowDown':
                e.preventDefault();
                nextId = currentMember.arrows.down;
                break;
            case 'ArrowLeft':
                e.preventDefault();
                nextId = currentMember.arrows.left;
                break;
            case 'ArrowRight':
                e.preventDefault();
                nextId = currentMember.arrows.right;
                break;
        }
        
        if (nextId && nextId !== '') {
            currentFocus = nextId;
            renderFamilyTree(familyData, totalHeight);
            centerOnFocus(familyData, totalHeight);
            highlightNuclearFamily(familyData, parentMapping, currentFocus);
        }
    });
}



// function initializeVisualization(familyData, parentMapping) {
//     // Check if mobile device before doing any visualization
//     if (isMobileDevice()) {
//         handleMobileAccess();
//         return;
//     }
    
//     let maxX = 0;
//     let maxY = 0;
    
//     Object.values(familyData).forEach(member => {
//         maxX = Math.max(maxX, member.x + BOX_WIDTH + 100);
//         maxY = Math.max(maxY, member.y + member.box_height + 100);
//     });
    
//     container.style.width = `${maxX}px`;
//     container.style.height = `${maxY}px`;
    
//     window.familyData = familyData;
//     window.totalHeight = maxY;
//     window.parentMapping = parentMapping;
    
//     const connectionsLayer = createFamilyConnections(parentMapping, familyData, maxY);
//     container.appendChild(connectionsLayer);
    
//     renderFamilyTree(familyData, maxY);
//     setupNavigation(familyData, parentMapping, maxY);
//     createZoomControls();
//     setupSearch();
    
//     centerOnFocus(familyData, maxY);
//     highlightNuclearFamily(familyData, parentMapping, currentFocus);
// }

// // Initialize when DOM is loaded
// // document.addEventListener('DOMContentLoaded', loadFamilyData);

// // Initialize when DOM is loaded
// document.addEventListener('DOMContentLoaded', () => {
//     handleMobileAccess(); // Check for mobile device immediately
//     loadFamilyData();
// });

// Info Modal
function createInfoButton() {
    const infoControls = document.createElement('div');
    infoControls.className = 'info-controls';
    infoControls.innerHTML = `
        <button id="infoBtn" class="zoom-btn">i</button>
    `;
    document.body.appendChild(infoControls);
}

function setupInfoModal() {
    const infoBtn = document.getElementById('infoBtn');
    const infoModal = document.getElementById('infoModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const closeBtn = document.getElementById('closeModal');

    if (!infoBtn || !infoModal || !modalOverlay || !closeBtn) {
        console.error('Modal elements not found:', { infoBtn, infoModal, modalOverlay, closeBtn });
        return;
    }

    // Initially hide the modal and overlay
    infoModal.style.display = 'none';
    modalOverlay.style.display = 'none';

    // Show modal when info button is clicked
    infoBtn.addEventListener('click', () => {
        console.log('Info button clicked');
        infoModal.style.display = 'block';
        modalOverlay.style.display = 'block';
    });

    // Hide modal when close button is clicked
    closeBtn.addEventListener('click', () => {
        infoModal.style.display = 'none';
        modalOverlay.style.display = 'none';
    });

    // Hide modal when clicking overlay
    modalOverlay.addEventListener('click', () => {
        infoModal.style.display = 'none';
        modalOverlay.style.display = 'none';
    });

    // Hide modal when pressing Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && infoModal.style.display === 'block') {
            infoModal.style.display = 'none';
            modalOverlay.style.display = 'none';
        }
    });
}

function initializeVisualization(familyData, parentMapping) {
    // Check for mobile device
    if (isMobileDevice()) {
        handleMobileAccess();
        ensureVisibility();  // Add this
        return;
    }
    
    // Calculate dimensions
    let maxX = 0;
    let maxY = 0;
    
    Object.values(familyData).forEach(member => {
        maxX = Math.max(maxX, member.x + BOX_WIDTH + 100);
        maxY = Math.max(maxY, member.y + member.box_height + 100);
    });
    
    // Set container dimensions
    container.style.width = `${maxX}px`;
    container.style.height = `${maxY}px`;
    
    // Store data in window object for global access
    window.familyData = familyData;
    window.totalHeight = maxY;
    window.parentMapping = parentMapping;
    
    // Create and append family connections
    const connectionsLayer = createFamilyConnections(parentMapping, familyData, maxY);
    container.appendChild(connectionsLayer);
    
    // Initialize components
    renderFamilyTree(familyData, maxY);
    setupNavigation(familyData, parentMapping, maxY);
    createZoomControls();
    createInfoButton();
    setupSearch();
    
    // Center on initial focus
    centerOnFocus(familyData, maxY);
    highlightNuclearFamily(familyData, parentMapping, currentFocus);

    // Setup info modal with a slight delay to ensure DOM elements are ready
    setTimeout(setupInfoModal, 100);
    
    // Ensure visibility after initialization
    ensureVisibility();  // Add this
}

// // Main initialization function
// function initializeVisualization(familyData, parentMapping) {
//     // Check for mobile device
//     if (isMobileDevice()) {
//         handleMobileAccess();
//         return;
//     }
    
//     // Calculate dimensions
//     let maxX = 0;
//     let maxY = 0;
    
//     Object.values(familyData).forEach(member => {
//         maxX = Math.max(maxX, member.x + BOX_WIDTH + 100);
//         maxY = Math.max(maxY, member.y + member.box_height + 100);
//     });
    
//     // Set container dimensions
//     container.style.width = `${maxX}px`;
//     container.style.height = `${maxY}px`;
    
//     // Store data in window object for global access
//     window.familyData = familyData;
//     window.totalHeight = maxY;
//     window.parentMapping = parentMapping;
    
//     // Create and append family connections
//     const connectionsLayer = createFamilyConnections(parentMapping, familyData, maxY);
//     container.appendChild(connectionsLayer);
    
//     // Initialize components
//     renderFamilyTree(familyData, maxY);
//     setupNavigation(familyData, parentMapping, maxY);
//     createZoomControls();
//     createInfoButton();
//     setupSearch();
    
//     // Center on initial focus
//     centerOnFocus(familyData, maxY);
//     highlightNuclearFamily(familyData, parentMapping, currentFocus);

//     // Setup info modal with a slight delay to ensure DOM elements are ready
//     setTimeout(setupInfoModal, 100);
// }

// Handle help banner display
function setupHelpBanner() {
    const helpBanner = document.getElementById('helpBanner');
    if (helpBanner) {
        // Show help banner initially
        helpBanner.style.display = 'block';

        // Hide help banner after 5 seconds
        setTimeout(() => {
            helpBanner.style.opacity = '0';
            setTimeout(() => {
                helpBanner.style.display = 'none';
            }, 500); // Wait for fade out animation
        }, 5000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Check for mobile device first
    if (handleMobileAccess()) {
        ensureVisibility();
        return;
    }

    // Setup help banner
    setupHelpBanner();

    // Load family data
    loadFamilyData();

    // Error handling for missing elements
    if (!container) {
        console.error('Tree container not found');
        ensureVisibility();
        return;
    }

    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (window.familyData && window.totalHeight) {
                centerOnFocus(window.familyData, window.totalHeight);
            }
            if (isMobileDevice()) {
                handleMobileAccess();
            }
        }, 250);
    });

    // Add visibility fallbacks
    window.addEventListener('load', ensureVisibility);
    window.addEventListener('error', ensureVisibility);
});

// document.addEventListener('DOMContentLoaded', () => {
//     // Check for mobile device first
//     if (handleMobileAccess()) {
//         ensureVisibility();  // Add this
//         return;
//     }

//     // Setup help banner
//     setupHelpBanner();

//     // Load family data
//     loadFamilyData();

//     // Error handling for missing elements
//     if (!container) {
//         console.error('Tree container not found');
//         ensureVisibility();  // Add this
//         return;
//     }

//     // Handle window resize
//     let resizeTimeout;
//     window.addEventListener('resize', () => {
//         clearTimeout(resizeTimeout);
//         resizeTimeout = setTimeout(() => {
//             if (window.familyData && window.totalHeight) {
//                 centerOnFocus(window.familyData, window.totalHeight);
//             }
//             if (isMobileDevice()) {
//                 handleMobileAccess();
//             }
//         }, 250);
//     });

//     // Add these event listeners for visibility fallback
//     window.addEventListener('error', ensureVisibility);
//     window.addEventListener('load', ensureVisibility);

//     // Handle errors gracefully
//     window.addEventListener('error', (e) => {
//         console.error('Application error:', e.error);
//         ensureVisibility();  // Add this
//     });
// });

// // Initialize everything when DOM is loaded
// document.addEventListener('DOMContentLoaded', () => {
//     // Check for mobile device first
//     if (handleMobileAccess()) {
//         return; // Stop initialization if mobile device
//     }

//     // Setup help banner
//     setupHelpBanner();

//     // Load family data
//     loadFamilyData();

//     // Error handling for missing elements
//     if (!container) {
//         console.error('Tree container not found');
//         return;
//     }

//     // Handle window resize
//     let resizeTimeout;
//     window.addEventListener('resize', () => {
//         clearTimeout(resizeTimeout);
//         resizeTimeout = setTimeout(() => {
//             if (window.familyData && window.totalHeight) {
//                 centerOnFocus(window.familyData, window.totalHeight);
//             }
//             if (isMobileDevice()) {
//                 handleMobileAccess();
//             }
//         }, 250);
//     });

//     // Handle errors gracefully
//     window.addEventListener('error', (e) => {
//         console.error('Application error:', e.error);
//         // You could add user-friendly error handling here
//     });
// });

// Export necessary functions if using modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeVisualization,
        handleMobileAccess,
        isMobileDevice,
        loadFamilyData
    };
}

setTimeout(ensureVisibility, 2000);

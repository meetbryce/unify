import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
//import { IMainMenu } from '@jupyterlab/application';
//import { Message } from '@lumino/messaging';

import { Widget } from '@lumino/widgets';
//import { Tree } from 'react-arborist';
//import { ReactWidget } from '@jupyterlab/apputils';

/*
interface APODResponse {
  copyright: string;
  date: string;
  explanation: string;
  media_type: 'video' | 'image';
  title: string;
  url: string;
};*/

/**
 * Initialization data for the unifyextension extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyterlab-apod',
  autoStart: true,
  requires: [ICommandPalette],
  activate: async (app: JupyterFrontEnd, palette: ICommandPalette) => {
    console.log('JupyterLab extension jupyterlab_apod is activated!');

    // Create a blank content widget inside of a MainAreaWidget
    const content = new Widget();
    const widget = new MainAreaWidget({ content });
    widget.id = 'apod-jupyterlab';
    widget.title.label = 'Unify Schema';
    widget.title.closable = true;
  
    // Add an image element to the content
    let img = document.createElement('img');
    content.node.appendChild(img);

    // Get a random date string in YYYY-MM-DD format
    function randomDate() {
      const start = new Date(2010, 1, 1);
      const end = new Date();
      const randomDate = new Date(start.getTime() + Math.random()*(end.getTime() - start.getTime()));
      return randomDate.toISOString().slice(0, 10);
    }
    randomDate();

    // Fetch info about a random picture
    /*
    const response = await fetch(`https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY`);
    //fetch(`https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&date=${randomDate()}`);
    const data = await response.json() as APODResponse;

    if (data.media_type === 'image') {
      // Populate the image
      img.src = data.url;
      img.title = data.title;
    } else {
      console.log('Random APOD was not a picture.');
    }*/

    // Add an application command
    const command: string = 'apod:open';
    app.commands.addCommand(command, {
      label: 'Unify Schema Browser',
      execute: () => {
        if (!widget.isAttached) {
          // Attach the widget to the main work area if it's not there
          app.shell.add(widget, 'left', {rank: 600});
        }
        // Activate the widget
        app.shell.activateById(widget.id);
      }
    });
    app.contextMenu.addItem({
      command: command,
      selector: '*'
    });
  
    // Add the command to the palette.
    palette.addItem({ command, category: 'Tutorial' });
  }
};

export default plugin;

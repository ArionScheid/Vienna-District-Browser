import {Component} from '@angular/core';
import {DataService} from './data.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
})
export class AppComponent {
  readonly repoUrl = 'https://github.com/ArionScheid/Vienna-District-Browser';
  readonly meta$ = this.data.meta();

  constructor(private data: DataService) {
  }
}
